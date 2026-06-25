import time

import numpy as np
import pytest
import sparse
from conftest import CHUNK_LEN, FS, N_CH, make_sparse_event_msg
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.event.binned import BinnedEventAggregator, BinnedEventAggregatorSettings


def test_event_rate_binned():
    dur = 1.1
    fs = 30_000.0
    chunk_dur = 0.1
    bin_dur = 0.03
    nchans = 128
    chunk_len = int(fs * chunk_dur)
    nchunk = int(dur / chunk_dur)

    rng = np.random.default_rng()
    s = sparse.random((int(fs * dur), nchans), density=0.0001, random_state=rng) > 0

    in_msgs = [
        AxisArray(
            data=s[chunk_ix * chunk_len : (chunk_ix + 1) * chunk_len],
            dims=["time", "ch"],
            axes={
                "time": AxisArray.Axis.TimeAxis(fs=fs, offset=chunk_ix * chunk_dur),
            },
        )
        for chunk_ix in range(nchunk)
    ]

    proc = BinnedEventAggregator(settings=BinnedEventAggregatorSettings(bin_duration=bin_dur))

    # Calculate the first message which sometimes takes longer due to initialization
    out_msgs = [proc(in_msgs[0])]

    # Make sure the first output message has the correct shape
    assert out_msgs[0].data.shape[0] == int(chunk_dur / bin_dur)

    # Calculate the remaining messages within perf_counters and assert they are processed quickly
    t_start = time.perf_counter()
    out_msgs.extend([proc(in_msg) for in_msg in in_msgs[1:]])
    t_elapsed = time.perf_counter() - t_start
    assert len(out_msgs) == nchunk
    assert t_elapsed < 0.5 * (dur - chunk_dur)  # Ensure processing is fast enough

    # Calculate the expected output and assert correctness
    n_binnable = int(dur / bin_dur)
    samps_per_bin = int(bin_dur * fs)
    expected = s[: n_binnable * samps_per_bin].reshape((n_binnable, samps_per_bin, -1)).sum(axis=1)
    stacked = AxisArray.concatenate(*out_msgs, dim="time")
    assert stacked.data.shape == expected.shape
    assert np.array_equal(stacked.data, expected.todense() / bin_dur)


def test_binned_event_aggregator_empty_time_after_init():
    """Normal → empty → normal: mid-stream empty message."""
    proc = BinnedEventAggregator(settings=BinnedEventAggregatorSettings(bin_duration=0.02))

    msg1 = make_sparse_event_msg(CHUNK_LEN, offset=0.0)
    msg_empty = make_sparse_event_msg(0, offset=CHUNK_LEN / FS)
    msg2 = make_sparse_event_msg(CHUNK_LEN, offset=CHUNK_LEN / FS)

    out1 = proc(msg1)
    assert out1.data.ndim == 2

    out_empty = proc(msg_empty)
    assert out_empty.data.ndim == 2

    out2 = proc(msg2)
    assert out2.data.ndim == 2
    assert out2.data.shape[1] == N_CH


def test_binned_event_aggregator_empty_time_first():
    """Empty → normal: empty first message triggers _reset_state on empty data."""
    proc = BinnedEventAggregator(settings=BinnedEventAggregatorSettings(bin_duration=0.02))

    msg_empty = make_sparse_event_msg(0, offset=0.0)
    msg_normal = make_sparse_event_msg(CHUNK_LEN, offset=0.0)

    out_empty = proc(msg_empty)
    assert out_empty.data.ndim == 2

    out_normal = proc(msg_normal)
    assert out_normal.data.ndim == 2
    assert out_normal.data.shape[1] == N_CH


def _sparse_chunks(spk: np.ndarray, fs: float, block: int) -> list[AxisArray]:
    out = []
    for start in range(0, spk.shape[0], block):
        out.append(
            AxisArray(
                data=sparse.COO.from_numpy(spk[start : start + block]),
                dims=["time", "ch"],
                axes={"time": AxisArray.TimeAxis(fs=fs, offset=start / fs)},
            )
        )
    return out


def _run(proc, msgs):
    return [r for r in (proc(m) for m in msgs) if r.data.size]


@pytest.mark.parametrize("fs", [30_000.0, 30_012.0])
def test_fractional_grid_offnominal(fs: float):
    """At an off-nominal rate the fractional binner stays on the nominal-gain
    wall-clock grid (gain == bin_duration, n_bins == int(n / (bin_duration*fs))),
    which is what aligns it with the spike-band-power branch. The legacy
    sample-locked path (Window) would instead report gain int(bin*fs)/fs."""
    bin_dur = 0.02
    n = 300_000
    spk = (np.random.default_rng(0).random((n, N_CH)) < 0.01).astype(float)

    proc = BinnedEventAggregator(settings=BinnedEventAggregatorSettings(bin_duration=bin_dur))
    out = _run(proc, _sparse_chunks(spk, fs, 7000))

    spb = bin_dur * fs
    assert sum(m.data.shape[0] for m in out) == int(n / spb)
    assert out[0].axes["time"].gain == pytest.approx(bin_dur)


@pytest.mark.parametrize("fs", [30_000.0, 30_012.0])
def test_chunk_invariance_offnominal(fs: float):
    """Output is identical regardless of how the stream is chunked."""
    bin_dur = 0.02
    spk = (np.random.default_rng(1).random((120_000, N_CH)) < 0.02).astype(float)

    whole = np.concatenate(
        [m.data for m in _run(BinnedEventAggregator(settings=BinnedEventAggregatorSettings(bin_duration=bin_dur)),
                              _sparse_chunks(spk, fs, 120_000))],
        axis=0,
    )
    frag = np.concatenate(
        [m.data for m in _run(BinnedEventAggregator(settings=BinnedEventAggregatorSettings(bin_duration=bin_dur)),
                              _sparse_chunks(spk, fs, 137))],
        axis=0,
    )
    assert whole.shape == frag.shape
    np.testing.assert_array_equal(whole, frag)


def test_count_vs_rate_scaling():
    """scale_output divides counts by bin_duration; otherwise raw counts."""
    fs = 30_000.0
    bin_dur = 0.02
    spk = (np.random.default_rng(2).random((60_000, N_CH)) < 0.02).astype(float)
    msgs = _sparse_chunks(spk, fs, 60_000)

    counts = _run(
        BinnedEventAggregator(settings=BinnedEventAggregatorSettings(bin_duration=bin_dur, scale_output=False)), msgs
    )[0]
    rate = _run(
        BinnedEventAggregator(settings=BinnedEventAggregatorSettings(bin_duration=bin_dur, scale_output=True)), msgs
    )[0]

    # Counts are exact integers; rate is counts / bin_duration.
    assert np.allclose(counts.data, np.round(counts.data))
    np.testing.assert_allclose(rate.data, counts.data / bin_dur, rtol=0, atol=1e-9)


def test_scale_by_value_weights_by_stored_value():
    """scale_by_value=True sums each event's stored value per bin; the default
    counts nonzero entries (1 each)."""
    fs = 30_000.0
    bin_dur = 0.02  # 600 samples/bin (exact at this fs)
    spb = int(bin_dur * fs)
    n = 3 * spb

    # One channel; known events with distinct values: two in bin 0, one in bin 1.
    values = np.zeros((n, 1), dtype=float)
    values[10, 0] = 2.0
    values[20, 0] = 3.0  # bin 0 -> count 2, value sum 5.0
    values[spb + 5, 0] = 7.0  # bin 1 -> count 1, value sum 7.0
    msg = AxisArray(
        data=sparse.COO.from_numpy(values),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=0.0)},
    )

    count = BinnedEventAggregator(
        settings=BinnedEventAggregatorSettings(bin_duration=bin_dur, scale_output=False, scale_by_value=False)
    )(msg)
    weighted = BinnedEventAggregator(
        settings=BinnedEventAggregatorSettings(bin_duration=bin_dur, scale_output=False, scale_by_value=True)
    )(msg)

    np.testing.assert_array_equal(count.data[:, 0], [2.0, 1.0, 0.0])
    np.testing.assert_allclose(weighted.data[:, 0], [5.0, 7.0, 0.0])


@pytest.mark.parametrize("fs", [30_000.0, 30_012.0])
def test_fractional_false_sample_locked(fs: float):
    """fractional=False bins a fixed round(bin_duration*fs) sample count, so the
    output gain is round(bin*fs)/fs (sample-locked) -- at an off-nominal rate this
    differs from the nominal bin_duration gain of the fractional grid."""
    bin_dur = 0.02
    n = 300_000
    spk = (np.random.default_rng(3).random((n, N_CH)) < 0.01).astype(float)

    proc = BinnedEventAggregator(settings=BinnedEventAggregatorSettings(bin_duration=bin_dur, fractional=False))
    out = _run(proc, _sparse_chunks(spk, fs, 7000))

    spb = round(bin_dur * fs)
    assert sum(m.data.shape[0] for m in out) == n // spb
    assert out[0].axes["time"].gain == pytest.approx(spb / fs)
