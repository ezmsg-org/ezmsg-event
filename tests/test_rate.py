import numpy as np
import sparse
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.event.rate import event_rate


def test_event_rate():
    dur = 1.0
    fs = 30_0000.0
    chunk_dur = 0.1
    bin_dur = 0.05
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
            key="test_event_rate",
        )
        for chunk_ix in range(nchunk)
    ]

    proc = event_rate(bin_duration=bin_dur)

    out_msgs = [proc.send(in_msg) for in_msg in in_msgs]
    assert len(out_msgs) == nchunk
    # Note: bin_dur != chunk_dur so we expect a different number of bins (len time axis) than chunks
    for om_ix, om in enumerate(out_msgs):
        assert om.key == "test_event_rate"
        assert om.dims == ["time", "ch"]
        assert om.data.shape == (
            int(chunk_dur / bin_dur),
            nchans,
        )  # Only works if even multiple
        assert om.axes["time"].gain == bin_dur
        assert om.axes["time"].offset == om_ix * chunk_dur

    stack = AxisArray.concatenate(*out_msgs, dim="time")
    expected = (
        np.sum(s.todense().reshape(-1, int(fs * bin_dur), nchans), axis=1) / bin_dur
    )
    assert stack.data.shape == expected.shape
    assert np.allclose(stack.data, expected)
