import numpy as np
import pytest
import sparse
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.event.window import windowing


@pytest.mark.parametrize("win_dur", [0.3, 1.0])
@pytest.mark.parametrize("win_shift", [None, 0.2, 1.0])
@pytest.mark.parametrize("zero_pad", ["input", "shift", "none"])
def test_sparse_window(
    win_dur: float,
    win_shift: float | None,
    zero_pad: str,
):
    fs = 100.0
    rng = np.random.default_rng()
    s = sparse.random((1_000, 5), density=0.1, random_state=rng) > 0
    in_msgs = [
        AxisArray(
            data=s[msg_ix * 100 : (msg_ix + 1) * 100],
            dims=["time", "ch"],
            axes={
                "time": AxisArray.Axis.TimeAxis(fs=fs, offset=msg_ix / fs),
            },
            key="test_sparse_window",
        )
        for msg_ix in range(10)
    ]

    proc = windowing(
        axis="time",
        newaxis="win",
        window_dur=win_dur,
        window_shift=win_shift,
        zero_pad_until=zero_pad,
    )
    out_msgs = [proc.send(_) for _ in in_msgs]
    for om in out_msgs:
        assert om.dims == ["win", "time", "ch"]
    print(out_msgs[0])
