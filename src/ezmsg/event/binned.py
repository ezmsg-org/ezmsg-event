import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, replace


class BinnedEventAggregatorSettings(ez.Settings):
    bin_duration: float = 0.05
    """
    Duration of each bin in seconds.
    This is the time interval over which events will be counted.
    """

    scale_output: bool = True
    """
    If True, the output will be scaled by the bin duration.
    This is useful for converting counts to rates.
    """

    axis: str = "time"


@processor_state
class BinnedEventAggregatorState:
    n_overflow: int = 0
    counts_in_overflow: npt.NDArray[np.int64] | None = None


class BinnedEventAggregator(
    BaseStatefulTransformer[BinnedEventAggregatorSettings, AxisArray, AxisArray, BinnedEventAggregatorState]
):
    def _hash_message(self, message: AxisArray) -> int:
        targ_ax_idx = message.get_axis_idx(self.settings.axis)
        non_targ_dims = message.dims[:targ_ax_idx] + message.dims[targ_ax_idx + 1 :]
        return hash(tuple(non_targ_dims))

    def _reset_state(self, message: AxisArray) -> None:
        self._state.n_overflow = 0
        targ_axis_idx = message.get_axis_idx(self.settings.axis)
        buff_shape = message.data.shape[:targ_axis_idx] + message.data.shape[targ_axis_idx + 1 :]
        self._state.counts_in_overflow = np.zeros(buff_shape, dtype=np.int64)

    def _process(self, message: AxisArray) -> AxisArray:
        # Quick maths
        targ_ax_idx = message.get_axis_idx(self.settings.axis)
        targ_axis = message.axes[self.settings.axis]
        samples_per_bin = int(self.settings.bin_duration * (1 / targ_axis.gain))

        # We will be slicing the data several times, so create a variable to hold the slices
        var_slice = [slice(None)] * message.data.ndim

        # Store for later use
        n_prev_overflow = self._state.n_overflow

        if self._state.n_overflow > 0:
            # Calculate how many samples from the input msg we can fit into the first bin,
            # given the current overflow state
            n_first = samples_per_bin - self._state.n_overflow
            # Sum the number of samples in the first bin then add to self._state.counts_in_overflow
            var_slice[targ_ax_idx] = slice(0, n_first)
            first_bin_counts = message.data[tuple(var_slice)].sum(axis=targ_ax_idx).todense()
            first_bin_counts += self._state.counts_in_overflow
        else:
            n_first = 0
            first_bin_counts = self._state.counts_in_overflow
            assert np.all(first_bin_counts == 0), "Overflow state should be zeroed out from previous iteration."

        # Calculate how many samples remain after the first bin
        n_remaining = message.data.shape[targ_ax_idx] - n_first
        n_full_bins = int(n_remaining / samples_per_bin)

        # Slice the n_first:-next_overflow samples into a segment that divides evenly into bins
        split_idx = n_first + n_full_bins * samples_per_bin
        var_slice[targ_ax_idx] = slice(n_first, split_idx)
        full_bins_data = message.data[tuple(var_slice)]

        # Reshape and sum for full bins
        new_shape = (
            full_bins_data.shape[:targ_ax_idx]
            + (n_full_bins, samples_per_bin)
            + full_bins_data.shape[targ_ax_idx + 1 :]
        )
        middle_bin_counts = full_bins_data.reshape(new_shape).sum(axis=targ_ax_idx + 1).todense()

        # Prepare output
        if self._state.n_overflow > 0:
            first_bin_counts = first_bin_counts.reshape(
                first_bin_counts.shape[:targ_ax_idx] + (1,) + first_bin_counts.shape[targ_ax_idx:]
            )
            output_data = np.concatenate([first_bin_counts, middle_bin_counts], axis=targ_ax_idx)
        else:
            output_data = middle_bin_counts

        if self.settings.scale_output:
            output_data = output_data / self.settings.bin_duration

        # Create the new output axis
        # For the target axis, backup the offset by the number of samples in the overflow
        out_axis = replace(
            targ_axis,
            gain=targ_axis.gain * samples_per_bin,
            offset=targ_axis.offset - n_prev_overflow * targ_axis.gain,
        )
        out_msg = replace(
            message,
            data=output_data,
            axes={k: v if k != self.settings.axis else out_axis for k, v in message.axes.items()},
        )

        # Calculate and store the overflow state.
        var_slice[targ_ax_idx] = slice(split_idx, None)
        overflow_data = message.data[tuple(var_slice)]
        self._state.n_overflow = overflow_data.shape[targ_ax_idx]
        self._state.counts_in_overflow = overflow_data.sum(axis=targ_ax_idx).todense()

        return out_msg


class BinnedEventAggregatorUnit(
    BaseTransformerUnit[BinnedEventAggregatorSettings, AxisArray, AxisArray, BinnedEventAggregator]
):
    SETTINGS = BinnedEventAggregatorSettings
