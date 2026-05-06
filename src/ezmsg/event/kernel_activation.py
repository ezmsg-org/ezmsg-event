"""
Compute binned kernel activation from events.

This module provides efficient computation of kernel-convolved features
at a lower output rate than the input. For exponential and alpha kernels,
uses a state-based approach that is O(n_events + n_bins) instead of
O(n_samples).

Input may be either ``sparse.COO`` (the typical output of
:class:`ezmsg.event.peak.ThresholdCrossingTransformer` in default mode) or a
dense array (from the same transformer with ``output_format=DENSE``). When the
input is dense and the configuration is COUNT + SUM (the rate-computation
case), the binning runs on the input's array namespace and stays on device
(e.g., MLX, CuPy). Other configurations with dense input fall back to
event-extraction and use the same code path as sparse input.
"""

from enum import Enum

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import sparse
from array_api_compat import get_namespace, is_numpy_array
from ezmsg.baseproc import BaseStatefulTransformer, BaseTransformerUnit, processor_state
from ezmsg.util.messages.axisarray import AxisArray, replace


class ActivationKernelType(str, Enum):
    """Supported kernel types for efficient binned activation."""

    EXPONENTIAL = "exponential"
    """Exponential decay: k(t) = exp(-t/tau) for t >= 0."""

    ALPHA = "alpha"
    """Alpha function: k(t) = (t/tau) * exp(-t/tau) for t >= 0."""

    COUNT = "count"
    """Simple event counting (no kernel, just count events per bin)."""


class BinAggregation(str, Enum):
    """How to aggregate activation within each bin."""

    LAST = "last"
    """Use activation value at end of bin (default for activation features)."""

    MEAN = "mean"
    """Average activation over the bin."""

    SUM = "sum"
    """Sum of activation over the bin (for count, this gives total count)."""

    MAX = "max"
    """Maximum activation in the bin."""


class BinnedKernelActivationSettings(ez.Settings):
    """Settings for BinnedKernelActivation."""

    kernel_type: ActivationKernelType = ActivationKernelType.EXPONENTIAL
    """Type of kernel to apply."""

    tau: float = 0.050
    """Time constant in seconds. For exponential: decay rate. For alpha: peak time."""

    bin_duration: float = 0.020
    """Output bin duration in seconds."""

    aggregation: BinAggregation = BinAggregation.LAST
    """How to aggregate activation within each bin."""

    scale_by_value: bool = False
    """If True, weight each event by its value. If False, all events contribute 1."""

    normalize: bool = True
    """If True, normalize kernel so integral equals 1."""

    rate_normalize: bool = False
    """If True, divide output by bin_duration to get events/second (for COUNT kernel)."""


@processor_state
class BinnedKernelActivationState:
    """State for BinnedKernelActivation."""

    # Current activation level per channel (for exponential/alpha)
    activation: npt.NDArray[np.float64] | None = None

    # For alpha kernel: auxiliary state variable
    alpha_aux: npt.NDArray[np.float64] | None = None

    # Time (in samples) since last state update per channel
    samples_since_update: npt.NDArray[np.int64] | None = None

    # Input sample rate (cached from first message)
    fs: float | None = None

    # Accumulated fractional bin samples for proper bin alignment
    bin_accumulator: float = 0.0


class BinnedKernelActivation(
    BaseStatefulTransformer[
        BinnedKernelActivationSettings,
        AxisArray,
        AxisArray,
        BinnedKernelActivationState,
    ]
):
    """
    Compute binned kernel activation from sparse events.

    For exponential and alpha kernels, uses an efficient state-based algorithm:
    - Exponential: activation[t] = sum_i exp(-(t - t_i) / tau)
    - Alpha: activation[t] = sum_i (t - t_i) / tau * exp(-(t - t_i) / tau)

    The algorithm only computes at event times and bin boundaries, giving
    O(n_events + n_bins) complexity instead of O(n_samples).

    Input: AxisArray with sparse.COO data (event times and values)
    Output: AxisArray with dense binned activation features

    Features:
        - Efficient for sparse events (much faster than dense convolution)
        - Handles chunk boundaries seamlessly
        - Supports exponential, alpha, and count kernels
        - Configurable bin aggregation (last, mean, sum, max)
    """

    def _hash_message(self, message: AxisArray) -> int:
        n_channels = message.data.shape[message.get_axis_idx("ch")] if "ch" in message.dims else 1
        if "time" not in message.axes or not hasattr(message.axes["time"], "gain"):
            raise ValueError("Could not determine sample rate from input message")
        # str(dtype) works for numpy ('bool', 'float32', ...) and mlx (which doesn't expose dtype.kind).
        return hash((message.data.ndim, str(message.data.dtype), n_channels, message.axes["time"].gain))

    def _reset_state(self, message: AxisArray) -> None:
        """Initialize state for new input stream."""
        n_channels = message.data.shape[message.get_axis_idx("ch")] if "ch" in message.dims else 1

        self._state.activation = np.zeros(n_channels, dtype=np.float64)
        self._state.samples_since_update = np.zeros(n_channels, dtype=np.int64)
        self._state.bin_accumulator = 0.0

        # For alpha kernel, we need auxiliary state
        if self.settings.kernel_type == ActivationKernelType.ALPHA:
            self._state.alpha_aux = np.zeros(n_channels, dtype=np.float64)

        # Cache sample rate -- we know time is in axes because _hash_message would raise an error otherwise
        time_axis = message.axes["time"]
        if time_axis.gain > 0:
            self._state.fs = 1.0 / time_axis.gain

    def _decay_to_sample(self, channel: int, target_sample: int) -> None:
        """
        Decay activation state to target sample.

        Uses the appropriate decay formula based on kernel type.
        """
        dt = target_sample - self._state.samples_since_update[channel]
        if dt <= 0:
            return

        tau_samples = self.settings.tau * self._state.fs
        decay = np.exp(-dt / tau_samples)

        if self.settings.kernel_type == ActivationKernelType.EXPONENTIAL:
            self._state.activation[channel] *= decay

        elif self.settings.kernel_type == ActivationKernelType.ALPHA:
            # Alpha kernel state update:
            # activation = sum of (t - t_i) / tau * exp(-(t - t_i) / tau)
            # We track: aux = sum of exp(-(t - t_i) / tau)
            #           activation = (derivative relationship)
            # Update: aux *= decay, activation = activation * decay + aux * dt / tau
            aux = self._state.alpha_aux[channel]
            self._state.alpha_aux[channel] = aux * decay
            # For alpha: d(activation)/dt = aux/tau - activation/tau
            # Integrated: activation(t+dt) = activation(t)*decay + aux*(1-decay)
            self._state.activation[channel] = self._state.activation[channel] * decay + aux * (1 - decay)

        self._state.samples_since_update[channel] = target_sample

    def _add_event(self, channel: int, sample: int, value: float) -> None:
        """Add an event contribution to the state."""
        # First decay to event time
        self._decay_to_sample(channel, sample)

        weight = value if self.settings.scale_by_value else 1.0
        if self.settings.normalize:
            # Normalize so integral equals 1
            weight /= self.settings.tau * self._state.fs

        if self.settings.kernel_type == ActivationKernelType.EXPONENTIAL:
            self._state.activation[channel] += weight

        elif self.settings.kernel_type == ActivationKernelType.ALPHA:
            # For alpha kernel, event adds to auxiliary state
            self._state.alpha_aux[channel] += weight

        elif self.settings.kernel_type == ActivationKernelType.COUNT:
            self._state.activation[channel] += weight

    def _get_activation_at_sample(self, channel: int, sample: int) -> float:
        """Get activation value at a specific sample."""
        self._decay_to_sample(channel, sample)
        return self._state.activation[channel]

    def _process(self, message: AxisArray) -> AxisArray:
        """Compute binned activation from sparse or dense event input.

        Dispatch:
            - Dense input + COUNT + SUM: fast path that stays on the input's array
              namespace (e.g., MLX, CuPy on device).
            - Dense input + any other config: extract events from non-zero entries
              and use the same code path as sparse input.
            - Sparse input: existing event-based path.
        """
        data = message.data
        is_sparse_input = isinstance(data, sparse.SparseArray)

        if not is_sparse_input:
            if (
                self.settings.kernel_type == ActivationKernelType.COUNT
                and self.settings.aggregation == BinAggregation.SUM
            ):
                return self._process_dense_count_sum(message)
            # Fall back: convert dense to sparse so the existing event-based path can run.
            data_np = data if is_numpy_array(data) else np.asarray(data)
            message = replace(message, data=sparse.COO.from_numpy(data_np))

        return self._process_events(message)

    def _process_events(self, message: AxisArray) -> AxisArray:
        """Compute binned activation from sparse events."""
        sparse_data = message.data
        n_samples = sparse_data.shape[0]
        n_channels = sparse_data.shape[1] if sparse_data.ndim > 1 else 1

        # Calculate bin parameters
        samples_per_bin = self.settings.bin_duration * self._state.fs
        total_samples = n_samples + self._state.bin_accumulator
        n_bins = int(total_samples / samples_per_bin)

        if n_bins == 0:
            # Not enough samples for a full bin yet
            self._state.bin_accumulator = total_samples

            # Still need to process events to update state
            if hasattr(sparse_data, "coords") and hasattr(sparse_data, "data"):
                coords = sparse_data.coords
                values = sparse_data.data
                for event_idx in range(len(values)):
                    sample_idx = int(coords[0, event_idx])
                    channel_idx = int(coords[1, event_idx]) if coords.shape[0] > 1 else 0
                    value = float(values[event_idx])
                    self._add_event(channel_idx, sample_idx, value)

            # Return empty output
            return replace(
                message,
                data=np.zeros((0, n_channels), dtype=np.float64),
                axes={
                    **message.axes,
                    "time": replace(message.axes["time"], gain=self.settings.bin_duration),
                },
            )

        # Calculate bin boundaries (in input samples, relative to chunk start)
        # Account for accumulator from previous chunk
        accumulator_before = self._state.bin_accumulator  # Save for offset calculation
        first_bin_end = samples_per_bin - self._state.bin_accumulator
        bin_ends = first_bin_end + np.arange(n_bins) * samples_per_bin

        # Update accumulator for next chunk
        self._state.bin_accumulator = total_samples - n_bins * samples_per_bin

        # Collect events sorted by time
        events = []
        if hasattr(sparse_data, "coords") and hasattr(sparse_data, "data"):
            coords = sparse_data.coords
            values = sparse_data.data
            for event_idx in range(len(values)):
                sample_idx = int(coords[0, event_idx])
                channel_idx = int(coords[1, event_idx]) if coords.shape[0] > 1 else 0
                value = float(values[event_idx])
                events.append((sample_idx, channel_idx, value))

        # Sort events by time
        events.sort(key=lambda x: x[0])

        # Process events and compute bin outputs
        output = np.zeros((n_bins, n_channels), dtype=np.float64)
        event_idx = 0

        if self.settings.aggregation == BinAggregation.LAST:
            # For LAST aggregation, process events up to each bin end
            for bin_idx, bin_end in enumerate(bin_ends):
                bin_end_sample = int(bin_end)

                # Process all events up to this bin end
                while event_idx < len(events) and events[event_idx][0] < bin_end_sample:
                    sample, channel, value = events[event_idx]
                    self._add_event(channel, sample, value)
                    event_idx += 1

                # Record activation at bin end for each channel
                for ch in range(n_channels):
                    output[bin_idx, ch] = self._get_activation_at_sample(ch, bin_end_sample)

        elif self.settings.aggregation == BinAggregation.SUM:
            # For SUM, accumulate within each bin
            # For COUNT type, include accumulated counts from previous partial bin
            for bin_idx, bin_end in enumerate(bin_ends):
                bin_end_sample = int(bin_end)

                # Start with any accumulated counts from previous chunk (for COUNT type)
                if bin_idx == 0 and self.settings.kernel_type == ActivationKernelType.COUNT:
                    bin_sum = self._state.activation.copy()
                    # Reset state for next bin accumulation
                    self._state.activation = np.zeros(n_channels, dtype=np.float64)
                else:
                    bin_sum = np.zeros(n_channels, dtype=np.float64)

                # Sum events within this bin
                while event_idx < len(events) and events[event_idx][0] < bin_end_sample:
                    sample, channel, value = events[event_idx]
                    weight = value if self.settings.scale_by_value else 1.0
                    bin_sum[channel] += weight
                    event_idx += 1

                output[bin_idx] = bin_sum

        elif self.settings.aggregation == BinAggregation.MEAN:
            # For MEAN with kernel, we'd need to integrate activation over bin
            # Approximate with samples at bin start and end
            bin_start = 0
            for bin_idx, bin_end in enumerate(bin_ends):
                bin_end_sample = int(bin_end)

                # Process events up to bin end
                while event_idx < len(events) and events[event_idx][0] < bin_end_sample:
                    sample, channel, value = events[event_idx]
                    self._add_event(channel, sample, value)
                    event_idx += 1

                # For exponential kernel, mean over [t0, t1] can be computed analytically
                # For simplicity, use midpoint approximation
                midpoint = (bin_start + bin_end_sample) // 2
                for ch in range(n_channels):
                    output[bin_idx, ch] = self._get_activation_at_sample(ch, midpoint)

                bin_start = bin_end_sample

        # Process any remaining events (for state continuity)
        while event_idx < len(events):
            sample, channel, value = events[event_idx]
            self._add_event(channel, sample, value)
            event_idx += 1

        # Update state sample counters relative to next chunk
        self._state.samples_since_update -= n_samples

        # Apply rate normalization if requested (divide by bin_duration to get events/second)
        if self.settings.rate_normalize:
            output = output / self.settings.bin_duration

        # Calculate output time offset
        # The first bin starts at (input_offset - accumulator_time)
        input_offset = message.axes["time"].offset if "time" in message.axes else 0.0
        accumulator_time = accumulator_before / self._state.fs
        output_offset = input_offset - accumulator_time

        return replace(
            message,
            data=output,
            axes={
                **message.axes,
                "time": AxisArray.TimeAxis(
                    fs=1.0 / self.settings.bin_duration,
                    offset=output_offset,
                ),
            },
        )

    def _process_dense_count_sum(self, message: AxisArray) -> AxisArray:
        """Fast path: dense input + COUNT kernel + SUM aggregation.

        Bins are summed using cumulative-sum arithmetic in the input's array
        namespace, so accelerator-resident inputs (MLX, CuPy) stay on device.
        Carry-over for the partial bin spanning chunk boundaries is held in
        ``state.activation`` (numpy) and shuttled across boundaries.
        """
        xp = get_namespace(message.data)
        data = message.data
        n_samples = data.shape[0]
        feature_shape = tuple(data.shape[1:])

        samples_per_bin = self.settings.bin_duration * self._state.fs
        accumulator_before = self._state.bin_accumulator
        total_samples = n_samples + accumulator_before
        n_bins = int(total_samples / samples_per_bin)

        # Per-sample contribution: 1 per non-zero, or the value itself if scaling.
        # Use the .astype() method form so the same call works for both numpy and mlx
        # (mlx.core has no top-level astype).
        if n_samples == 0:
            contrib = xp.zeros((0,) + feature_shape, dtype=xp.float32)
        elif self.settings.scale_by_value:
            contrib = data.astype(xp.float32)
        else:
            contrib = (data != 0).astype(xp.float32)

        # Pull state into the input namespace for on-device math.
        overflow_xp = xp.asarray(self._state.activation.reshape(feature_shape)).astype(xp.float32)

        if n_bins == 0:
            # No complete bins this chunk — accumulate everything into the carry-over.
            new_overflow = overflow_xp + (xp.sum(contrib, axis=0) if n_samples > 0 else overflow_xp * 0)
            self._state.activation = np.asarray(new_overflow).reshape(self._state.activation.shape)
            self._state.bin_accumulator = total_samples
            return replace(
                message,
                data=xp.zeros((0,) + feature_shape, dtype=xp.float32),
                axes={
                    **message.axes,
                    "time": replace(message.axes["time"], gain=self.settings.bin_duration),
                },
            )

        # Bin boundaries (in input-sample space, integer-truncated as in the event-based path).
        first_bin_end = samples_per_bin - accumulator_before
        bin_ends_float = first_bin_end + np.arange(n_bins) * samples_per_bin
        bin_end_samples = bin_ends_float.astype(np.int64)
        bin_start_samples = np.concatenate(([np.int64(0)], bin_end_samples[:-1]))

        # Cumulative sum, prepended with zeros so cumsum_padded[k] = sum(contrib[:k]).
        # Use cumsum (in both numpy and mlx); numpy via array_api_compat also exposes
        # the standard `cumulative_sum`, but mlx does not.
        cumsum = xp.cumsum(contrib, axis=0)
        zero_row = xp.zeros((1,) + feature_shape, dtype=cumsum.dtype)
        cumsum_padded = xp.concat((zero_row, cumsum), axis=0)

        end_idx = xp.asarray(bin_end_samples)
        start_idx = xp.asarray(bin_start_samples)
        bin_sums = xp.take(cumsum_padded, end_idx, axis=0) - xp.take(cumsum_padded, start_idx, axis=0)

        # Add carry-over from the previous chunk's partial bin into bin 0.
        overflow_pad_first = overflow_xp[None, ...]
        if n_bins > 1:
            overflow_pad_rest = xp.zeros((n_bins - 1,) + feature_shape, dtype=bin_sums.dtype)
            overflow_pad = xp.concat((overflow_pad_first, overflow_pad_rest), axis=0)
        else:
            overflow_pad = overflow_pad_first
        output = bin_sums + overflow_pad

        # New carry-over: events past the last complete bin remain in the partial bin.
        last_bin_end = int(bin_end_samples[-1])
        if last_bin_end < n_samples:
            new_overflow = xp.sum(contrib[last_bin_end:], axis=0)
        else:
            new_overflow = xp.zeros(feature_shape, dtype=cumsum.dtype)
        self._state.activation = np.asarray(new_overflow).reshape(self._state.activation.shape)
        self._state.bin_accumulator = total_samples - n_bins * samples_per_bin

        if self.settings.rate_normalize:
            output = output / self.settings.bin_duration

        accumulator_time = accumulator_before / self._state.fs
        input_offset = message.axes["time"].offset if "time" in message.axes else 0.0
        output_offset = input_offset - accumulator_time

        return replace(
            message,
            data=output,
            axes={
                **message.axes,
                "time": AxisArray.TimeAxis(
                    fs=1.0 / self.settings.bin_duration,
                    offset=output_offset,
                ),
            },
        )


class BinnedKernelActivationUnit(
    BaseTransformerUnit[
        BinnedKernelActivationSettings,
        AxisArray,
        AxisArray,
        BinnedKernelActivation,
    ]
):
    """Unit for BinnedKernelActivation."""

    SETTINGS = BinnedKernelActivationSettings
