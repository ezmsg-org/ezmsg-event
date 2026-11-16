"""The stream-metadata contract: ``config_id`` on every output, and empty messages.

A stage whose output layout differs from its input (sparse events to dense bins, dense signal
to sparse events, a new sample rate) stamps an id derived from the input's; a declared change of
the input's id resets it; an empty message passes through.
"""

import numpy as np
import pytest
import sparse
from ezmsg.baseproc import end_of_stream
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis, compute_config_id
from ezmsg.util.messages.util import replace

from ezmsg.event.binned import BinnedEventAggregator, BinnedEventAggregatorSettings
from ezmsg.event.kernel import ArrayKernel
from ezmsg.event.kernel_activation import (
    ActivationKernelType,
    BinnedKernelActivation,
    BinnedKernelActivationSettings,
)
from ezmsg.event.kernel_insert import SparseKernelInserter, SparseKernelInserterSettings
from ezmsg.event.peak import OutputFormat, ThresholdCrossingTransformer, ThresholdSettings
from ezmsg.event.poissonevents import PoissonEventSettings, PoissonEventTransformer
from ezmsg.event.rate import EventRateSettings, Rate
from ezmsg.event.refractory import RefractorySettings, RefractoryTransformer

FS = 1000.0
LABELS = ["a", "b"]


def _declare(data, offset=0.0, segment=0, fs=FS, declared=True) -> AxisArray:
    msg = AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=offset), "ch": CoordinateAxis(np.array(LABELS), dims=["ch"])},
        key="dev",
        stream_dim="time",
        segment=segment,
    )
    return replace(msg, config_id=compute_config_id(msg)) if declared else msg


def _events(n=100, seed=0, **kw) -> AxisArray:
    data = sparse.random((n, len(LABELS)), density=0.05, random_state=np.random.default_rng(seed)) > 0
    return _declare(data, **kw)


def _signal(n=100, seed=0, **kw) -> AxisArray:
    return _declare(np.random.default_rng(seed).standard_normal((n, len(LABELS))), **kw)


def _rates(n=5, seed=0, **kw) -> AxisArray:
    return _declare(np.full((n, len(LABELS)), 50.0), fs=50.0, **kw)


# (factory, input maker, whether the output layout differs from the input's)
PROCESSORS = {
    "kernel_activation": (lambda: BinnedKernelActivation(BinnedKernelActivationSettings(bin_duration=0.02)), _events),
    "rate": (lambda: Rate(EventRateSettings(bin_duration=0.02)), _events),
    "binned": (lambda: BinnedEventAggregator(BinnedEventAggregatorSettings(bin_duration=0.02)), _events),
    "kernel_insert": (
        lambda: SparseKernelInserter(SparseKernelInserterSettings(kernel=ArrayKernel(np.ones(5)))),
        _events,
    ),
    "threshold": (lambda: ThresholdCrossingTransformer(ThresholdSettings(threshold=1.0)), _signal),
    "threshold_dense": (
        lambda: ThresholdCrossingTransformer(ThresholdSettings(threshold=1.0, output_format=OutputFormat.DENSE)),
        _signal,
    ),
    "threshold_peak_val": (
        lambda: ThresholdCrossingTransformer(
            ThresholdSettings(threshold=1.0, align_on_peak=True, return_peak_val=True)
        ),
        _signal,
    ),
    "poisson": (lambda: PoissonEventTransformer(PoissonEventSettings(output_fs=1000.0)), _rates),
}


@pytest.mark.parametrize("name", PROCESSORS)
def test_a_reshaping_stage_stamps_a_derived_config_id(name):
    make_proc, make_msg = PROCESSORS[name]
    proc = make_proc()
    first = make_msg()
    out1 = proc(first)
    out2 = proc(replace(make_msg(seed=1), axes={**first.axes, "time": replace(first.axes["time"], offset=1.0)}))

    assert out1.config_id not in (0, first.config_id)
    assert out2.config_id == out1.config_id
    assert out1.segment == first.segment
    # The id says what it describes: the same layout from a fresh stage digests the same.
    assert make_proc()(first).config_id == out1.config_id
    # A different input configuration gives a different output id.
    other = replace(first, key="other")
    assert make_proc()(replace(other, config_id=compute_config_id(other))).config_id != out1.config_id


@pytest.mark.parametrize("name", PROCESSORS)
def test_an_undeclared_input_gives_an_undeclared_output(name):
    make_proc, make_msg = PROCESSORS[name]
    assert make_proc()(make_msg(declared=False)).config_id == 0


def test_refractory_keeps_the_input_config_id():
    msg = _events()
    assert RefractoryTransformer(RefractorySettings(dur=0.01))(msg).config_id == msg.config_id


@pytest.mark.parametrize("name", [*PROCESSORS, "refractory"])
def test_an_end_of_stream_message_passes_through(name):
    if name == "refractory":
        make_proc, make_msg = (lambda: RefractoryTransformer(RefractorySettings(dur=0.01))), _events
    else:
        make_proc, make_msg = PROCESSORS[name]
    proc = make_proc()
    first = make_msg()
    out = proc(first)
    marker = end_of_stream(first)
    at_marker = proc(marker)
    assert at_marker.segment == marker.segment
    assert at_marker.data.shape[at_marker.get_axis_idx("time")] == 0
    assert at_marker.config_id == out.config_id
    # And data resumes in a later segment.
    resumed = proc(replace(make_msg(seed=2), segment=marker.segment + 1))
    assert resumed.segment == marker.segment + 1


class TestKernelActivationResetsOnADeclaredChange:
    """With its `_hash_message` override gone, a declared config_id change is what resets it."""

    @staticmethod
    def _proc():
        return BinnedKernelActivation(
            BinnedKernelActivationSettings(kernel_type=ActivationKernelType.EXPONENTIAL, tau=0.05, bin_duration=0.02)
        )

    @staticmethod
    def _late_events():
        """Events on the last samples, so a carried activation would show in the next bins."""
        coords = np.array([[95, 96, 97, 98, 99] * 2, [0] * 5 + [1] * 5])
        return _declare(sparse.COO(coords, np.ones(10, dtype=bool), shape=(100, 2)))

    def _carried_equals_fresh(self, nxt):
        assert nxt.config_id != self._late_events().config_id
        proc = self._proc()
        proc(self._late_events())
        np.testing.assert_array_equal(proc(nxt).data, self._proc()(nxt).data)

    def test_sparse_to_dense(self):
        dense = np.zeros((100, 2))
        dense[10, 0] = 1.0
        self._carried_equals_fresh(_declare(dense, offset=0.1))

    def test_dtype(self):
        coords = np.array([[10], [0]])
        self._carried_equals_fresh(_declare(sparse.COO(coords, np.ones(1, dtype=np.float32), shape=(100, 2)), 0.1))

    def test_unchanged_config_keeps_the_activation(self):
        proc = self._proc()
        proc(self._late_events())
        nxt = replace(_events(seed=3), axes={**_events().axes, "time": AxisArray.TimeAxis(fs=FS, offset=0.1)})
        assert not np.array_equal(proc(nxt).data, self._proc()(nxt).data)


class TestKernelInserterTail:
    @staticmethod
    def _proc():
        return SparseKernelInserter(SparseKernelInserterSettings(kernel=ArrayKernel(np.arange(1.0, 6.0))))

    @staticmethod
    def _at(sample, n=10, offset=0.0, segment=0):
        coords = np.array([[sample], [0]]) if sample is not None else np.zeros((2, 0), dtype=int)
        values = np.ones(coords.shape[1], dtype=int)
        return _declare(sparse.COO(coords, values, shape=(n, 2)), offset=offset, segment=segment)

    def test_short_and_empty_chunks_keep_the_tail(self):
        """A chunk shorter than the tail (here empty, then one sample) used to drop the rest."""
        proc = self._proc()
        outs = [
            proc(self._at(8)),
            proc(self._at(None, n=0, offset=0.010)),
            proc(self._at(None, n=1, offset=0.010)),
            proc(self._at(None, n=10, offset=0.011)),
        ]
        full = np.concatenate([o.data[:, 0] for o in outs])
        np.testing.assert_array_equal(full[8:13], [1.0, 2.0, 3.0, 4.0, 5.0])
        assert full.sum() == 15.0
