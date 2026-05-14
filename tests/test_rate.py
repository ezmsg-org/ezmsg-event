import time

import numpy as np
import pytest
import sparse
from conftest import CHUNK_LEN, FS, N_CH, make_sparse_event_msg
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.event.peak import OutputFormat, ThresholdCrossingTransformer
from ezmsg.event.rate import EventRateSettings, Rate
from ezmsg.event.util.simulate import generate_white_noise_with_events


def test_event_rate_composite():
    dur = 1.0
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

    proc = Rate(settings=EventRateSettings(bin_duration=bin_dur))

    out_msgs = [proc(in_msgs[0])]

    assert out_msgs[0].data.shape[0] == int(chunk_dur / bin_dur)

    # Calculate the remaining messages within perf_counters and assert they are processed quickly
    t_start = time.perf_counter()
    out_msgs.extend([proc(in_msg) for in_msg in in_msgs[1:]])
    t_elapsed = time.perf_counter() - t_start
    assert len(out_msgs) == nchunk
    _ = t_elapsed < (dur - chunk_dur)  # Ensure processing is fast enough

    n_bins_seen = 0
    for om_ix, om in enumerate(out_msgs):
        assert om.dims == ["time", "ch"]
        assert np.isclose(om.axes["time"].gain, bin_dur)
        assert np.isclose(om.axes["time"].offset, n_bins_seen * bin_dur)
        n_bins_seen += om.shape[0]

    stack = AxisArray.concatenate(*out_msgs, dim="time")
    t_proc = n_bins_seen * bin_dur
    samp_proc = int(t_proc * fs)
    s_proc = s[:samp_proc].todense().reshape(-1, int(fs * bin_dur), nchans)
    expected = np.sum(s_proc, axis=1) / bin_dur
    assert stack.data.shape == expected.shape
    assert np.allclose(stack.data, expected)


def test_rate_empty_time_after_init():
    """Normal → empty → normal: mid-stream empty message."""
    proc = Rate(EventRateSettings(bin_duration=0.02))

    msg1 = make_sparse_event_msg(CHUNK_LEN, offset=0.0)
    msg_empty = make_sparse_event_msg(0, offset=CHUNK_LEN / FS)
    msg2 = make_sparse_event_msg(CHUNK_LEN, offset=CHUNK_LEN / FS)

    out1 = proc(msg1)
    assert out1.data.ndim == 2

    out_empty = proc(msg_empty)
    assert out_empty.data.shape[0] == 0 or out_empty.data.ndim == 2

    out2 = proc(msg2)
    assert out2.data.ndim == 2
    assert out2.data.shape[1] == N_CH


def _make_msg(arr: np.ndarray, fs: float, offset: float) -> AxisArray:
    return AxisArray(
        data=arr,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=offset),
            "ch": AxisArray.CoordinateAxis(data=np.arange(arr.shape[1]), dims=["ch"]),
        },
    )


def _run_threshold_rate(
    arr_chunks, *, fs: float, threshold: float, refrac_dur: float, bin_duration: float, output_format: OutputFormat
) -> list[AxisArray]:
    thresh = ThresholdCrossingTransformer(
        threshold=threshold,
        refrac_dur=refrac_dur,
        output_format=output_format,
    )
    rate = Rate(EventRateSettings(bin_duration=bin_duration))
    out, samp_off = [], 0
    for chunk in arr_chunks:
        out.append(rate(thresh(_make_msg(chunk, fs, samp_off / fs))))
        samp_off += chunk.shape[0]
    return out


@pytest.mark.parametrize(
    ("fs", "bin_duration", "refrac_dur"),
    [
        (1000.0, 0.010, 0.006),  # integer samples per bin
        (30012.0048, 0.020, 0.001),  # fractional samples per bin
    ],
)
def test_rate_dense_input_matches_sparse(fs: float, bin_duration: float, refrac_dur: float):
    """Rate fed dense ThresholdCrossing output must match the sparse pipeline bin-for-bin."""
    threshold = 2.5
    rate_range = (1, 100)
    n_chans = 8
    dur = 0.6

    chunk_lens = [int(fs * 0.05), int(fs * 0.13), int(fs * 0.21)]
    chunk_lens.append(int(fs * dur) - sum(chunk_lens))
    in_dat = generate_white_noise_with_events(fs, dur, n_chans, rate_range, 0.05, threshold)

    chunks, idx = [], 0
    for cl in chunk_lens:
        chunks.append(in_dat[idx : idx + cl])
        idx += cl

    sp = _run_threshold_rate(
        chunks,
        fs=fs,
        threshold=threshold,
        refrac_dur=refrac_dur,
        bin_duration=bin_duration,
        output_format=OutputFormat.SPARSE,
    )
    ds = _run_threshold_rate(
        chunks,
        fs=fs,
        threshold=threshold,
        refrac_dur=refrac_dur,
        bin_duration=bin_duration,
        output_format=OutputFormat.DENSE,
    )

    assert len(sp) == len(ds)
    for sp_msg, ds_msg in zip(sp, ds):
        assert sp_msg.data.shape == ds_msg.data.shape
        assert sp_msg.axes["time"].gain == ds_msg.axes["time"].gain
        assert np.isclose(sp_msg.axes["time"].offset, ds_msg.axes["time"].offset)
        np.testing.assert_allclose(sp_msg.data, ds_msg.data)


def test_rate_empty_time_first():
    """Empty → normal: empty first message triggers _reset_state on empty data."""
    proc = Rate(EventRateSettings(bin_duration=0.02))

    msg_empty = make_sparse_event_msg(0, offset=0.0)
    msg_normal = make_sparse_event_msg(CHUNK_LEN, offset=0.0)

    out_empty = proc(msg_empty)
    assert out_empty.data.ndim == 2

    out_normal = proc(msg_normal)
    assert out_normal.data.ndim == 2
    assert out_normal.data.shape[1] == N_CH
