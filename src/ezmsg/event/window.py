from dataclasses import replace
import typing

import numpy as np
import sparse
import ezmsg.core as ez
from ezmsg.sigproc.base import GenAxisArray
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import (
    AxisArray,
    slice_along_axis,
    sliding_win_oneaxis,
)

from .util.array import sliding_win_oneaxis as sparse_sliding_win


@consumer
def windowing(
    axis: str | None = None,
    newaxis: str = "win",
    window_dur: float | None = None,
    window_shift: float | None = None,
    zero_pad_until: str = "input",
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Construct a generator that yields windows of data from an input :obj:`AxisArray`.

    Args:
        axis: The axis along which to segment windows.
            If None, defaults to the first dimension of the first seen AxisArray.
        newaxis: New axis on which windows are delimited, immediately
        preceding the target windowed axis. The data length along newaxis may be 0 if
        this most recent push did not provide enough data for a new window.
        If window_shift is None then the newaxis length will always be 1.
        window_dur: The duration of the window in seconds.
            If None, the function acts as a passthrough and all other parameters are ignored.
        window_shift: The shift of the window in seconds.
            If None (default), windowing operates in "1:1 mode", where each input yields exactly one most-recent window.
        zero_pad_until: Determines how the function initializes the buffer.
            Can be one of "input" (default), "full", "shift", or "none". If `window_shift` is None then this field is
            ignored and "input" is always used.

            - "input" (default) initializes the buffer with the input then prepends with zeros to the window size.
              The first input will always yield at least one output.
            - "shift" fills the buffer until `window_shift`.
              No outputs will be yielded until at least `window_shift` data has been seen.
            - "none" does not pad the buffer. No outputs will be yielded until at least `window_dur` data has been seen.

    Returns:
        A (primed) generator that accepts .send(an AxisArray object) and yields a list of windowed
        AxisArray objects. The list will always be length-1 if `newaxis` is not None or `window_shift` is None.
    """
    # Check arguments
    if newaxis is None:
        ez.logger.warning("`newaxis` must not be None. Setting to 'win'.")
        newaxis = "win"
    if window_shift is None and zero_pad_until != "input":
        ez.logger.warning(
            "`zero_pad_until` must be 'input' if `window_shift` is None. "
            f"Ignoring received argument value: {zero_pad_until}"
        )
        zero_pad_until = "input"
    elif window_shift is not None and zero_pad_until == "input":
        ez.logger.warning(
            "windowing is non-deterministic with `zero_pad_until='input'` as it depends on the size "
            "of the first input. We recommend using 'shift' when `window_shift` is float-valued."
        )

    msg_out = AxisArray(np.array([]), dims=[""])

    # State variables
    buffer: sparse.SparseArray | None = None
    window_samples: int | None = None
    window_shift_samples: int | None = None
    shift_deficit: int = 0
    b_1to1 = window_shift is None
    newaxis_warned: bool = b_1to1
    out_newaxis: AxisArray.Axis | None = None
    out_dims: list[str] | None = None

    check_inputs = {"samp_shape": None, "fs": None, "key": None}

    while True:
        msg_in: AxisArray = yield msg_out

        if window_dur is None:
            msg_out = msg_in
            continue

        axis = axis or msg_in.dims[0]
        axis_idx = msg_in.get_axis_idx(axis)
        axis_info = msg_in.get_axis(axis)
        fs = 1.0 / axis_info.gain

        if not newaxis_warned and newaxis in msg_in.dims:
            ez.logger.warning(
                f"newaxis {newaxis} present in input dims. Using {newaxis}_win instead"
            )
            newaxis_warned = True
            newaxis = f"{newaxis}_win"

        samp_shape = msg_in.data.shape[:axis_idx] + msg_in.data.shape[axis_idx + 1 :]

        # If buffer unset or input stats changed, create a new buffer
        b_reset = buffer is None
        b_reset = b_reset or samp_shape != check_inputs["samp_shape"]
        b_reset = b_reset or fs != check_inputs["fs"]
        b_reset = b_reset or msg_in.key != check_inputs["key"]
        if b_reset:
            # Update check variables
            check_inputs["samp_shape"] = samp_shape
            check_inputs["fs"] = fs
            check_inputs["key"] = msg_in.key

            window_samples = int(window_dur * fs)
            if not b_1to1:
                window_shift_samples = int(window_shift * fs)
            if zero_pad_until == "none":
                req_samples = window_samples
            elif zero_pad_until == "shift" and not b_1to1:
                req_samples = window_shift_samples
            else:  # i.e. zero_pad_until == "input"
                req_samples = msg_in.data.shape[axis_idx]
            n_zero = max(0, window_samples - req_samples)
            buffer = sparse.zeros(
                msg_in.data.shape[:axis_idx]
                + (n_zero,)
                + msg_in.data.shape[axis_idx + 1 :]
            )

        # Add new data to buffer.
        buffer = sparse.concatenate((buffer, msg_in.data), axis=axis_idx)

        # Create a vector of buffer timestamps to track axis `offset` in output(s)
        buffer_offset = np.arange(buffer.shape[axis_idx]).astype(float)
        # Adjust so first _new_ sample at index 0
        buffer_offset -= buffer_offset[-msg_in.data.shape[axis_idx]]
        # Convert form indices to 'units' (probably seconds).
        buffer_offset *= axis_info.gain
        buffer_offset += axis_info.offset

        if not b_1to1 and shift_deficit > 0:
            n_skip = min(buffer.shape[axis_idx], shift_deficit)
            if n_skip > 0:
                buffer = slice_along_axis(buffer, slice(n_skip, None), axis_idx)
                buffer_offset = buffer_offset[n_skip:]
                shift_deficit -= n_skip

        # Prepare reusable parts of output
        if out_newaxis is None:
            out_dims = msg_in.dims[:axis_idx] + [newaxis] + msg_in.dims[axis_idx:]
            out_newaxis = replace(
                axis_info,
                gain=0.0 if b_1to1 else axis_info.gain * window_shift_samples,
                offset=0.0,  # offset modified per-msg below
            )

        # Generate outputs.
        # Preliminary copy of axes without the axes that we are modifying.
        out_axes = {k: v for k, v in msg_in.axes.items() if k not in [newaxis, axis]}

        # Update targeted (windowed) axis so that its offset is relative to the new axis
        # TODO: If we have `anchor_newest=True` then offset should be -win_dur
        out_axes[axis] = replace(axis_info, offset=0.0)

        # How we update .data and .axes[newaxis] depends on the windowing mode.
        if b_1to1:
            # one-to-one mode -- Each send yields exactly one window containing only the most recent samples.
            buffer = slice_along_axis(buffer, slice(-window_samples, None), axis_idx)
            out_dat = buffer.reshape(
                buffer.shape[:axis_idx] + (1,) + buffer.shape[axis_idx:]
            )
            out_newaxis = replace(out_newaxis, offset=buffer_offset[-window_samples])
        elif buffer.shape[axis_idx] >= window_samples:
            # Deterministic window shifts.
            out_dat = sparse_sliding_win(
                buffer, window_samples, axis_idx, step=window_shift_samples
            )
            offset_view = sliding_win_oneaxis(buffer_offset, window_samples, 0)[
                ::window_shift_samples
            ]
            out_newaxis = replace(out_newaxis, offset=offset_view[0, 0])

            # Drop expired beginning of buffer and update shift_deficit
            multi_shift = window_shift_samples * out_dat.shape[axis_idx]
            shift_deficit = max(0, multi_shift - buffer.shape[axis_idx])
            buffer = slice_along_axis(buffer, slice(multi_shift, None), axis_idx)
        else:
            # Not enough data to make a new window. Return empty data.
            empty_data_shape = (
                msg_in.data.shape[:axis_idx]
                + (0, window_samples)
                + msg_in.data.shape[axis_idx + 1 :]
            )
            out_dat = sparse.zeros(empty_data_shape, dtype=msg_in.data.dtype)
            # out_newaxis will have first timestamp in input... but mostly meaningless because output is size-zero.
            out_newaxis = replace(out_newaxis, offset=axis_info.offset)

        msg_out = replace(
            msg_in, data=out_dat, dims=out_dims, axes={**out_axes, newaxis: out_newaxis}
        )


class WindowSettings(ez.Settings):
    axis: str | None = None
    newaxis: str | None = None  # new axis for output. No new axes if None
    window_dur: float | None = None  # Sec. passthrough if None
    window_shift: float | None = None  # Sec. Use "1:1 mode" if None
    zero_pad_until: str = "full"  # "full", "shift", "input", "none"


class Window(GenAxisArray):
    """:obj:`Unit` for :obj:`bandpower`."""

    SETTINGS = WindowSettings

    def construct_generator(self):
        self.STATE.gen = windowing(
            axis=self.SETTINGS.axis,
            newaxis=self.SETTINGS.newaxis,
            window_dur=self.SETTINGS.window_dur,
            window_shift=self.SETTINGS.window_shift,
            zero_pad_until=self.SETTINGS.zero_pad_until,
        )
