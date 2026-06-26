"""Cross-package proof that EventRate and the ezmsg-sigproc dense binner share a grid.

Both now route their bin boundaries through ``ezmsg.sigproc.util.binning.BinSchedule``.
This test feeds identical data to the real ``EventRate`` (count/sum path) and to
``BinnedAggregateTransformer`` (operation=SUM) and asserts they produce the same
output time axis (gain + offset) and the same bin counts -- the alignment property
the shared primitive exists to guarantee -- including at an off-nominal sample
rate and under adversarial chunking.

EventRate has ``rate_normalize=True`` (counts / bin_duration), so its values equal
the SUM binner's values divided by bin_duration; everything else must match exactly.
"""

import numpy as np
import pytest
from ezmsg.sigproc.aggregate import AggregationFunction
from ezmsg.sigproc.binned_aggregate import BinnedAggregateTransformer
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.event.rate import EventRateSettings, Rate


def _make_msg(arr: np.ndarray, fs: float, offset: float) -> AxisArray:
    return AxisArray(
        data=arr,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=offset),
            "ch": AxisArray.CoordinateAxis(data=np.arange(arr.shape[1]), dims=["ch"]),
        },
    )


@pytest.mark.parametrize("fs", [1000.0, 30012.0, 30030.0])
@pytest.mark.parametrize("block_size", [50000, 1, 777])
def test_eventrate_and_dense_binner_share_grid(fs: float, block_size: int):
    bin_duration = 0.02
    rng = np.random.default_rng(0)
    # Dense spike train (0/1). EventRate's dense COUNT+SUM path counts non-zeros;
    # BinnedAggregate(SUM) sums them -- same quantity per bin.
    spikes = (rng.random((40000, 3)) < 0.01).astype(np.float32)

    rate = Rate(EventRateSettings(bin_duration=bin_duration))
    binner = BinnedAggregateTransformer(
        axis="time", bin_duration=bin_duration, operation=AggregationFunction.SUM, fractional=True
    )

    rate_out, binner_out, samp_off = [], [], 0
    for start in range(0, spikes.shape[0], block_size):
        chunk = spikes[start : start + block_size]
        offset = samp_off / fs
        rate_out.append(rate(_make_msg(chunk, fs, offset)))
        binner_out.append(binner(_make_msg(chunk, fs, offset)))
        samp_off += chunk.shape[0]

    # Per-message: identical gain + offset on every non-empty output, and matching
    # bin counts. This is exactly what a downstream Merge(align_axis="time") needs.
    for r, b in zip(rate_out, binner_out):
        assert r.data.shape[0] == b.data.shape[0]
        if r.data.shape[0] == 0:
            continue
        assert r.axes["time"].gain == pytest.approx(b.axes["time"].gain)
        assert r.axes["time"].offset == pytest.approx(b.axes["time"].offset)

    rate_all = np.concatenate([m.data for m in rate_out], axis=0)
    binner_all = np.concatenate([m.data for m in binner_out], axis=0)
    assert rate_all.shape == binner_all.shape
    # EventRate divides counts by bin_duration (rate_normalize); undo to compare.
    np.testing.assert_allclose(rate_all * bin_duration, binner_all, rtol=0, atol=1e-6)


@pytest.mark.parametrize("fs", [30012.0, 30030.0])
@pytest.mark.parametrize("block_size", [40000, 1, 333])
def test_eventrate_fractional_false_is_sample_locked(fs: float, block_size: int):
    """fractional=False EventRate bins on Window's int(bin_duration*fs) grid and
    shares it with BinnedAggregate(fractional=False). Its output rate is the
    sample-locked fs/int(bin_duration*fs), not the nominal 1/bin_duration."""
    bin_duration = 0.02
    window_samples = int(bin_duration * fs)
    expected_gain = window_samples / fs
    rng = np.random.default_rng(1)
    spikes = (rng.random((40000, 2)) < 0.01).astype(np.float32)

    rate = Rate(EventRateSettings(bin_duration=bin_duration, fractional=False))
    binner = BinnedAggregateTransformer(
        axis="time", bin_duration=bin_duration, operation=AggregationFunction.SUM, fractional=False
    )

    rate_out, binner_out, samp_off = [], [], 0
    for start in range(0, spikes.shape[0], block_size):
        chunk = spikes[start : start + block_size]
        offset = samp_off / fs
        rate_out.append(rate(_make_msg(chunk, fs, offset)))
        binner_out.append(binner(_make_msg(chunk, fs, offset)))
        samp_off += chunk.shape[0]

    for r, b in zip(rate_out, binner_out):
        assert r.data.shape[0] == b.data.shape[0]
        if r.data.shape[0] == 0:
            continue
        # Sample-locked gain (== Window's), not the nominal bin_duration.
        assert r.axes["time"].gain == pytest.approx(expected_gain)
        assert r.axes["time"].gain == pytest.approx(b.axes["time"].gain)
        assert r.axes["time"].offset == pytest.approx(b.axes["time"].offset)

    rate_all = np.concatenate([m.data for m in rate_out], axis=0)
    binner_all = np.concatenate([m.data for m in binner_out], axis=0)
    assert rate_all.shape == binner_all.shape
    # rate_normalize now divides by the actual bin duration (== expected_gain).
    np.testing.assert_allclose(rate_all * expected_gain, binner_all, rtol=0, atol=1e-6)
