import time

import numpy as np
import sparse
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.event.rate import EventRateSettings, Rate


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
