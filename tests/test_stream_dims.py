"""Every stage finds its dims from the message, not from the names ``"time"`` and ``"ch"``.

Each stage is fed the same data three ways: ``(time, ch)``, transposed to ``(ch, time)``, and
renamed to ``(t, unit)`` with ``stream_dim="t"``. The outputs must agree up to that
transposition and renaming.
"""

import numpy as np
import pytest
import sparse
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

from ezmsg.event.kernel import ArrayKernel
from ezmsg.event.kernel_activation import BinnedKernelActivation, BinnedKernelActivationSettings
from ezmsg.event.kernel_insert import SparseKernelInserter, SparseKernelInserterSettings
from ezmsg.event.peak import ThresholdCrossingTransformer, ThresholdSettings
from ezmsg.event.poissonevents import PoissonEventSettings, PoissonEventTransformer
from ezmsg.event.rate import EventRateSettings, Rate
from ezmsg.event.refractory import RefractorySettings, RefractoryTransformer

FS = 1000.0
N_CH = 3
# name -> (stream dim, channel dim, channel dim first); the stream dim is declared as stream_dim
LAYOUTS = {
    "conventional": ("time", "ch", False),
    "transposed": ("time", "ch", True),
    "renamed": ("t", "unit", False),
}


def _as(data: np.ndarray, layout: str, fs: float = FS, as_sparse: bool = False) -> AxisArray:
    time_dim, ch_dim, transpose = LAYOUTS[layout]
    data = data.T if transpose else data
    return AxisArray(
        data=sparse.COO.from_numpy(data) if as_sparse else data,
        dims=[ch_dim, time_dim] if transpose else [time_dim, ch_dim],
        axes={
            time_dim: AxisArray.TimeAxis(fs=fs, offset=0.5),
            ch_dim: CoordinateAxis(np.array([f"c{i}" for i in range(N_CH)]), dims=[ch_dim]),
        },
        key="dev",
        stream_dim=time_dim,
    )


def _events() -> np.ndarray:
    return (np.random.default_rng(0).random((250, N_CH)) < 0.05).astype(float)


def _signal() -> np.ndarray:
    return np.random.default_rng(1).standard_normal((250, N_CH)) * 2.0


# name -> (processor factory, input maker, output keeps the input's layout)
STAGES = {
    "kernel_activation": (
        lambda: BinnedKernelActivation(BinnedKernelActivationSettings(bin_duration=0.02)),
        lambda layout: _as(_events(), layout, as_sparse=True),
        True,
    ),
    "rate_dense": (
        lambda: Rate(EventRateSettings(bin_duration=0.02)),
        lambda layout: _as(_events(), layout),
        True,
    ),
    "kernel_insert": (
        lambda: SparseKernelInserter(SparseKernelInserterSettings(kernel=ArrayKernel(np.arange(1.0, 6.0)))),
        lambda layout: _as(_events(), layout, as_sparse=True),
        True,
    ),
    "threshold": (
        lambda: ThresholdCrossingTransformer(ThresholdSettings(threshold=2.0)),
        lambda layout: _as(_signal(), layout),
        False,  # puts the stream dim first, as it always has
    ),
    "threshold_scaled_peak_val": (
        lambda: ThresholdCrossingTransformer(
            ThresholdSettings(threshold=1.5, auto_scale_tau=0.05, align_on_peak=True, return_peak_val=True)
        ),
        lambda layout: _as(_signal(), layout),
        False,
    ),
    "refractory": (
        lambda: RefractoryTransformer(RefractorySettings(dur=0.01)),
        lambda layout: _as(_events(), layout, as_sparse=True),
        True,
    ),
}


def _conventional(out: AxisArray, layout: str) -> np.ndarray:
    """``out``'s data as a dense ``(time, ch)`` array, located by its dims."""
    time_dim = LAYOUTS[layout][0]
    data = out.data.todense() if isinstance(out.data, sparse.SparseArray) else np.asarray(out.data)
    return np.moveaxis(data, out.dims.index(time_dim), 0)


def _check_layout(out: AxisArray, msg: AxisArray, keeps_layout: bool, layout: str) -> None:
    time_dim = LAYOUTS[layout][0]
    assert out.stream_dim == time_dim
    assert out.dims == (msg.dims if keeps_layout else [time_dim] + [d for d in msg.dims if d != time_dim])
    assert out.data.shape[out.dims.index(time_dim)] == _conventional(out, layout).shape[0]


@pytest.mark.parametrize("layout", ["transposed", "renamed"])
@pytest.mark.parametrize("name", STAGES)
def test_output_matches_the_conventional_layout(name, layout):
    make_proc, make_msg, keeps_layout = STAGES[name]
    ref = make_proc()(make_msg("conventional"))
    msg = make_msg(layout)
    out = make_proc()(msg)

    _check_layout(out, msg, keeps_layout, layout)
    np.testing.assert_array_equal(_conventional(out, layout), _conventional(ref, "conventional"))
    time_dim = LAYOUTS[layout][0]
    assert out.axes[time_dim].gain == pytest.approx(ref.axes["time"].gain)
    assert out.axes[time_dim].offset == pytest.approx(ref.axes["time"].offset)


@pytest.mark.parametrize("layout", ["conventional", "transposed", "renamed"])
def test_poisson_events_land_on_the_right_channel(layout):
    """Random, so compared by where the events fall: only channel 1 has a rate."""
    rates = np.zeros((10, N_CH))
    rates[:, 1] = 500.0
    msg = _as(rates, layout, fs=50.0)
    out = PoissonEventTransformer(PoissonEventSettings(output_fs=FS))(msg)

    _check_layout(out, msg, keeps_layout=False, layout=layout)
    events = _conventional(out, layout)
    assert events.shape == (200, N_CH)
    assert events[:, 1].sum() > 0 and events[:, [0, 2]].sum() == 0
    time_dim = LAYOUTS[layout][0]
    assert out.axes[time_dim].gain == pytest.approx(1 / FS)
