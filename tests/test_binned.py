import time

import numpy as np
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
