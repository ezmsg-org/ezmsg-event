import time

import numpy as np
import sparse
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
