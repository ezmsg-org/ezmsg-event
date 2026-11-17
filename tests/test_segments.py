"""What each processor does with a message's ``segment`` and ``attrs``.

Both must reach the output, and the first message of a new segment must be processed as if by
a processor that had seen nothing before it: the samples either side of a gap are not
neighbours, so nothing carried from before it (a tail of samples, a refractory count, a
kernel's decaying activation, a partly filled bin) may leak across.
"""

import asyncio

import numpy as np
import pytest
import sparse
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis
from ezmsg.util.messages.util import replace

from ezmsg.event.binned import BinnedEventAggregator, BinnedEventAggregatorSettings, BinnedEventAggregatorUnit
from ezmsg.event.kernel import ArrayKernel
from ezmsg.event.kernel_activation import (
    ActivationKernelType,
    BinAggregation,
    BinnedKernelActivation,
    BinnedKernelActivationSettings,
    BinnedKernelActivationUnit,
)
from ezmsg.event.kernel_insert import SparseKernelInserter, SparseKernelInserterSettings
from ezmsg.event.peak import ThresholdCrossingTransformer, ThresholdSettings
from ezmsg.event.rate import EventRate, EventRateSettings, Rate
from ezmsg.event.refractory import RefractorySettings, RefractoryTransformer

FS = 30_000.0
N_TIME = 700  # not a whole number of 20 ms bins, so a partly filled bin is carried
LABELS = ["a", "b", "c"]
ATTRS = {"unit": "uV", "device": "sim"}
GAP = 5.0  # seconds between the end of one segment and the start of the next


def _msg(data, offset: float, segment: int) -> AxisArray:
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=FS, offset=offset), "ch": CoordinateAxis(np.array(LABELS), dims=["ch"])},
        key="dev",
        attrs=ATTRS,
        stream_dim="time",
        segment=segment,
    )


def _dense(seed: int, offset: float, segment: int) -> AxisArray:
    # A large step between the two segments, so that state carried across shows.
    data = np.random.default_rng(seed).standard_normal((N_TIME, len(LABELS))) + 50.0 * segment
    return _msg(data, offset, segment)


def _events(seed: int, offset: float, segment: int) -> AxisArray:
    data = sparse.random((N_TIME, len(LABELS)), density=0.05, random_state=np.random.default_rng(seed)) > 0
    return _msg(data, offset, segment)


def _pair(make):
    return make(1, 0.0, 0), make(2, N_TIME / FS + GAP, 1)


PROCESSORS = {
    "threshold": (lambda: ThresholdCrossingTransformer(ThresholdSettings(threshold=1.0)), _dense),
    "threshold_peak_val": (
        lambda: ThresholdCrossingTransformer(
            ThresholdSettings(threshold=1.0, align_on_peak=True, return_peak_val=True)
        ),
        _dense,
    ),
    "refractory": (lambda: RefractoryTransformer(RefractorySettings(dur=0.01)), _events),
    "kernel_insert": (
        lambda: SparseKernelInserter(SparseKernelInserterSettings(kernel=ArrayKernel(np.hanning(90)))),
        _events,
    ),
    "kernel_activation": (
        lambda: BinnedKernelActivation(
            BinnedKernelActivationSettings(kernel_type=ActivationKernelType.EXPONENTIAL, tau=0.05, bin_duration=0.02)
        ),
        _events,
    ),
    "rate": (lambda: Rate(EventRateSettings(bin_duration=0.02)), _events),
    "binned": (lambda: BinnedEventAggregator(BinnedEventAggregatorSettings(bin_duration=0.02)), _events),
}


def _dense_data(msg: AxisArray) -> np.ndarray:
    data = msg.data
    return data.todense() if isinstance(data, sparse.SparseArray) else np.asarray(data)


@pytest.mark.parametrize("name", PROCESSORS)
def test_segment_and_attrs_reach_the_output(name):
    make_proc, make_msg = PROCESSORS[name]
    proc = make_proc()
    for msg in _pair(make_msg):
        out = proc(msg)
        assert out.segment == msg.segment
        assert out.attrs == ATTRS


@pytest.mark.parametrize("name", PROCESSORS)
def test_a_new_segment_is_processed_as_if_nothing_came_before(name):
    make_proc, make_msg = PROCESSORS[name]
    first, second = _pair(make_msg)

    proc = make_proc()
    proc(first)
    continued = proc(second)
    fresh = make_proc()(second)

    np.testing.assert_array_equal(_dense_data(continued), _dense_data(fresh))
    assert continued.axes["time"] == fresh.axes["time"]

    if name.startswith("threshold"):
        return  # carries one sample back, which this data does not make visible
    # And the test can tell: without the segment change, what came before does leak in.
    proc = make_proc()
    proc(first)
    same_segment = proc(replace(second, segment=first.segment))
    leaked = _dense_data(same_segment).shape != _dense_data(fresh).shape or not np.array_equal(
        _dense_data(same_segment), _dense_data(fresh)
    )
    assert leaked or same_segment.axes["time"] != fresh.axes["time"]


def test_threshold_keeps_its_scale_estimate_across_a_segment():
    """The auto-scaler's running mean and variance describe the signal, which a gap in time
    does not change; restarting them would mis-scale the first samples after every gap. Only
    what belongs to the samples before the gap is dropped."""
    settings = ThresholdSettings(threshold=3.0, auto_scale_tau=0.5)
    rng = np.random.default_rng(0)

    def noise(offset, segment, scale=20.0):
        return _msg(rng.standard_normal((N_TIME, len(LABELS))) * scale, offset, segment)

    proc = ThresholdCrossingTransformer(settings)
    for i in range(30):
        proc(noise(i * N_TIME / FS, 0))
    scaler = proc.state.scaler

    # Ten times louder after the gap: against the scale already learned, most of it is over
    # threshold; against a scale learned from these samples alone, little is.
    after_gap = noise(30 * N_TIME / FS + GAP, 1, scale=200.0)
    continued = proc(after_gap)
    fresh = ThresholdCrossingTransformer(settings)(after_gap)

    assert proc.state.scaler is scaler
    assert _dense_data(continued).sum() > 3 * _dense_data(fresh).sum()
    assert continued.segment == 1


# --- At a gap, a partial bin is emitted only when the output is a rate. ---

RATE_STAGES = {
    "rate": lambda: Rate(EventRateSettings(bin_duration=0.02)),
    "binned": lambda: BinnedEventAggregator(BinnedEventAggregatorSettings(bin_duration=0.02, scale_output=True)),
    "kernel_activation": lambda: BinnedKernelActivation(
        BinnedKernelActivationSettings(
            kernel_type=ActivationKernelType.COUNT,
            aggregation=BinAggregation.SUM,
            bin_duration=0.02,
            rate_normalize=True,  # normalize=True (the default) must not touch a count
        )
    ),
}

COUNT_STAGES = {
    "binned_count": lambda: BinnedEventAggregator(BinnedEventAggregatorSettings(bin_duration=0.02, scale_output=False)),
    "kernel_activation_count": lambda: BinnedKernelActivation(
        BinnedKernelActivationSettings(
            kernel_type=ActivationKernelType.COUNT, aggregation=BinAggregation.SUM, bin_duration=0.02
        )
    ),
    "exponential_last": lambda: BinnedKernelActivation(BinnedKernelActivationSettings(bin_duration=0.02)),
    "alpha_mean": lambda: BinnedKernelActivation(
        BinnedKernelActivationSettings(
            kernel_type=ActivationKernelType.ALPHA, aggregation=BinAggregation.MEAN, bin_duration=0.02
        )
    ),
}

SPB = 600  # 20 ms at FS; N_TIME = 700 leaves a 100-sample partial bin


def _partial_bin_events(dense: bool, offset: float = 1.0, segment: int = 0) -> AxisArray:
    data = np.zeros((N_TIME, len(LABELS)))
    data[[10, 650, 660, 699], 0] = 1.0  # three in the partial bin
    data[[620], 1] = 1.0  # one
    return _msg(data if dense else sparse.COO.from_numpy(data), offset, segment)


@pytest.mark.parametrize("dense", [False, True], ids=["sparse", "dense"])
@pytest.mark.parametrize("name", RATE_STAGES)
def test_flush_emits_a_partial_rate_bin(name, dense):
    proc = RATE_STAGES[name]()
    msg = _partial_bin_events(dense)
    out = proc(msg)
    tail = proc.flush()

    partial_duration = (N_TIME - SPB) / FS
    np.testing.assert_allclose(tail.data, [[3 / partial_duration, 1 / partial_duration, 0.0]], rtol=1e-6)
    assert tail.axes["time"].offset == pytest.approx(1.0 + SPB / FS)
    assert tail.axes["time"].gain == pytest.approx(out.axes["time"].gain)
    assert (tail.segment, tail.config_id, tail.key, tail.attrs) == (msg.segment, out.config_id, msg.key, ATTRS)
    assert tail.dims == out.dims and tail.data.dtype == out.data.dtype
    assert proc.flush() is None


@pytest.mark.parametrize("name", RATE_STAGES)
def test_flush_with_a_final_message_appends_the_partial_bin(name):
    out = RATE_STAGES[name]().flush(_partial_bin_events(dense=False))
    assert out.data.shape == (2, len(LABELS))
    assert out.data[1, 0] == pytest.approx(3 / ((N_TIME - SPB) / FS))


@pytest.mark.parametrize("name", RATE_STAGES)
def test_flush_emits_nothing_for_an_empty_partial_bin(name):
    proc = RATE_STAGES[name]()
    proc(_msg(sparse.COO.from_numpy(np.zeros((2 * SPB, len(LABELS)))), 0.0, 0))
    assert proc.flush() is None


@pytest.mark.parametrize("name", COUNT_STAGES)
def test_flush_drops_a_partial_count_or_activation_bin(name):
    proc = COUNT_STAGES[name]()
    proc(_partial_bin_events(dense=False))
    assert proc.flush() is None
    # The dropped partial bin (e.g. a kernel MEAN's running sum) does not reach the next segment.
    nxt = _partial_bin_events(dense=False, offset=10.0, segment=1)
    np.testing.assert_array_equal(proc(nxt).data, COUNT_STAGES[name]()(nxt).data)


@pytest.mark.parametrize("name", RATE_STAGES)
def test_after_a_flush_the_next_segment_starts_clean(name):
    proc = RATE_STAGES[name]()
    proc(_partial_bin_events(dense=False))
    proc.flush()
    nxt = _partial_bin_events(dense=False, offset=10.0, segment=1)
    continued = proc(nxt)
    fresh = RATE_STAGES[name]()(nxt)
    np.testing.assert_array_equal(continued.data, fresh.data)
    assert continued.axes["time"] == fresh.axes["time"]
    # And the same after a flush within one segment (a config change): a fresh grid, nothing carried.
    proc.flush()
    again = proc(replace(nxt, axes={**nxt.axes, "time": AxisArray.TimeAxis(fs=FS, offset=20.0)}))
    np.testing.assert_array_equal(again.data, fresh.data)


@pytest.mark.parametrize("name", ["rate", "binned", "kernel_activation"])
def test_unit_publishes_the_partial_rate_bin_before_the_next_segment(name):
    unit_cls, settings = {
        "rate": (EventRate, EventRateSettings(bin_duration=0.02)),
        "binned": (BinnedEventAggregatorUnit, BinnedEventAggregatorSettings(bin_duration=0.02)),
        "kernel_activation": (
            BinnedKernelActivationUnit,
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.COUNT,
                aggregation=BinAggregation.SUM,
                bin_duration=0.02,
                rate_normalize=True,
            ),
        ),
    }[name]
    unit = unit_cls(settings)
    unit.create_processor()

    async def drive(msg):
        return [m async for _, m in unit.on_signal(msg)]

    asyncio.run(drive(_partial_bin_events(dense=False)))
    tail, first = asyncio.run(drive(_partial_bin_events(dense=False, offset=10.0, segment=1)))
    assert tail.segment == 0 and tail.data.shape == (1, len(LABELS))
    assert first.segment == 1 and first.data.shape == (1, len(LABELS))
