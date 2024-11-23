"""
Count number of events in a given time window. Optionally, divide by window duration to get rate.
"""

from dataclasses import replace
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import consumer
from ezmsg.sigproc.base import GenAxisArray

from .window import windowing


@consumer
def event_rate(
    bin_duration: float = 0.05,
) -> typing.Generator[AxisArray, AxisArray, None]:
    """


    Args:
        bin_duration:

    Returns:
        A primed generator object that yields an :obj:`AxisArray` object of event rates for every
        :obj:`AxisArray` of sparse events it receives via `send`.
    """
    msg_out = AxisArray(np.array([]), dims=[""])

    win_proc = windowing(
        axis="time",
        newaxis="win",
        window_dur=bin_duration,
        window_shift=bin_duration,
        zero_pad_until="none",
    )
    out_dims: list[str] | None = None
    out_axes: dict[str, AxisArray.Axis] | None = None

    while True:
        msg_in: AxisArray = yield msg_out

        win_msg = win_proc.send(msg_in)

        b_reset = out_dims is None
        if b_reset:
            # Fixup `dims`
            out_dims = list(win_msg.dims)
            out_dims.remove("time")
            out_dims[out_dims.index("win")] = "time"
            # Fixup axes
            out_axes = {k: v for k, v in win_msg.axes.items() if k != "time"}

        # Sum over time
        time_ax = win_msg.get_axis_idx("time")
        counts_per_bin = np.sum(win_msg.data, axis=time_ax)
        # Scale by 1 / bin_duration to get rates
        rates_per_bin = counts_per_bin / bin_duration
        # Densify
        rates_per_bin = rates_per_bin.todense()
        msg_out = replace(
            win_msg,
            data=rates_per_bin,
            dims=out_dims,
            axes={**out_axes, "time": win_msg.axes["win"]},
        )


class EventRateSettings(ez.Settings):
    bin_duration: float = 0.05


class EventRate(GenAxisArray):
    SETTINGS = EventRateSettings

    def construct_generator(self):
        self.STATE.gen = event_rate(
            bin_duration=self.SETTINGS.bin_duration,
        )
