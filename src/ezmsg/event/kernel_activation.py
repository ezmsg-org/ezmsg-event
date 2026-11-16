"""
Compute binned kernel activation from events.

This module provides efficient computation of kernel-convolved features
at a lower output rate than the input. For exponential and alpha kernels,
uses a state-based approach that is O(n_events + n_bins) instead of
O(n_samples).

Input may be either ``sparse.COO`` (the default output of
:class:`ezmsg.event.peak.ThresholdCrossingTransformer`) or a dense array from the
same transformer with ``output_format=DENSE``. When the input is dense and the
configuration is COUNT + SUM (the rate-computation case), the binning runs on the
input's array namespace and stays on device (e.g., MLX, CuPy). Other configurations
with dense input fall back to event extraction and use the same code path as sparse
input.
"""

import math
from enum import Enum

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import sparse
from array_api_compat import get_namespace, is_numpy_array
from ezmsg.baseproc import BaseStatefulTransformer, BaseTransformerUnit, processor_state
from ezmsg.sigproc.util.binning import BinSchedule
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
    """Maximum activation at any sample in the bin. Not defined for the COUNT kernel."""


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
    """If True, divide output by the bin's duration to get events/second (for COUNT
    kernel). The divisor is the *actual* bin duration (``samples_per_bin / fs``),
    which equals ``bin_duration`` exactly in fractional mode."""

    fractional: bool = True
    """If True (default), bins span a *fractional* ``bin_duration * fs`` samples
    with a carry accumulator; bins track the nominal duration and the output gain
    is exactly ``bin_duration``. If False, bins span a *fixed* ``int(bin_duration *
    fs)`` samples and the output gain is ``int(bin_duration * fs) / fs``
    (sample-locked, matching :obj:`ezmsg.sigproc.window.Window`)."""


@processor_state
class BinnedKernelActivationState:
    """State for BinnedKernelActivation."""

    # Current activation level per channel (for exponential/alpha)
    activation: npt.NDArray[np.float64] | None = None

    dense_carry: object | None = None
    """Partial-bin COUNT+SUM state in the dense input's array namespace."""

    # For alpha kernel: auxiliary state variable
    alpha_aux: npt.NDArray[np.float64] | None = None

    # Time (in samples) since last state update per channel
    samples_since_update: npt.NDArray[np.int64] | None = None

    # Input sample rate (cached from first message)
    fs: float | None = None

    # Shared bin-boundary schedule. Owns samples-per-bin, the fractional carry,
    # the global bin index, and the output gain/offset -- the same primitive the
    # ezmsg-sigproc dense binner uses, so both land on an identical grid.
    schedule: BinSchedule | None = None

    bin_max: npt.NDArray[np.float64] | None = None
    """MAX aggregation: the largest activation so far in the open bin, per channel."""

    bin_sum: npt.NDArray[np.float64] | None = None
    """SUM with a kernel: the activation summed over the open bin's samples so far, per channel."""


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
        if "time" not in message.axes or not hasattr(message.axes["time"], "gain"):
            raise ValueError("Could not determine sample rate from input message")
        # The base class covers the key, the dims, the channel count and the
        # sample rate. Two things it cannot see are added here:
        #   dtype   -- `str()` rather than `.kind`, which mlx arrays do not expose.
        #   backend -- the per-channel state is numpy either way, but the code
        #              paths that fill it differ between sparse, numpy and mlx.
        # Folding in the key is new, and is the point: `activation`,
        # `dense_carry` and `samples_since_update` are per-channel running state,
        # so carrying them into a different stream is the same failure as a
        # filter keeping the previous device's history.
        backend = "sparse" if isinstance(message.data, sparse.SparseArray) else get_namespace(message.data).__name__
        return self._message_hash(message, extra=(str(message.data.dtype), backend))

    def _reset_state(self, message: AxisArray) -> None:
        """Initialize state for new input stream."""
        if self.settings.kernel_type == ActivationKernelType.COUNT and self.settings.aggregation == BinAggregation.MAX:
            raise ValueError(
                "BinAggregation.MAX is not defined for the COUNT kernel: a count has no time course within a bin. "
                "Use BinAggregation.SUM for the number of events per bin."
            )
        n_channels = message.data.shape[message.get_axis_idx("ch")] if "ch" in message.dims else 1

        self._state.activation = np.zeros(n_channels, dtype=np.float64)
        self._state.dense_carry = None
        # -1: the state describes the instant before the first sample, so MAX sees sample 0 too.
        self._state.samples_since_update = np.full(n_channels, -1, dtype=np.int64)
        self._state.bin_max = np.full(n_channels, -np.inf)
        self._state.bin_sum = np.zeros(n_channels)

        # For alpha kernel, we need auxiliary state
        if self.settings.kernel_type == ActivationKernelType.ALPHA:
            self._state.alpha_aux = np.zeros(n_channels, dtype=np.float64)

        # Cache sample rate -- we know time is in axes because _hash_message would raise an error otherwise
        time_axis = message.axes["time"]
        if time_axis.gain > 0:
            self._state.fs = 1.0 / time_axis.gain

        # The schedule owns the boundary arithmetic from here on; fractional mode
        # tracks the nominal bin_duration, sample-locked mode matches Window's grid.
        self._state.schedule = BinSchedule(bin_duration=self.settings.bin_duration, fractional=self.settings.fractional)
        self._state.schedule.reset(self._state.fs)

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
            # For alpha: d(activation)/dt = aux/tau - activation/tau, with aux decaying too.
            # Integrated exactly: activation(t+dt) = (activation(t) + aux(t)*dt/tau) * decay,
            # so the result does not depend on how the interval is split.
            self._state.activation[channel] = (self._state.activation[channel] + aux * dt / tau_samples) * decay

        self._state.samples_since_update[channel] = target_sample

    def _add_event(self, channel: int, sample: int, value: float) -> None:
        """Add an event contribution to the state."""
        max_agg = self.settings.aggregation == BinAggregation.MAX
        if max_agg:
            self._track_max(channel, sample - 1)
        sum_agg = self._kernel_sum()
        if sum_agg:
            # Fold samples up to and including this one, before the event; its own k(0) is added below.
            self._track_sum(channel, sample)
        # First decay to event time
        self._decay_to_sample(channel, sample)

        weight = value if self.settings.scale_by_value else 1.0
        # A count has no kernel shape to normalise (and the SUM bins count raw weights).
        if self.settings.normalize and self.settings.kernel_type != ActivationKernelType.COUNT:
            # Normalize so integral equals 1
            weight /= self.settings.tau * self._state.fs

        if self.settings.kernel_type == ActivationKernelType.EXPONENTIAL:
            self._state.activation[channel] += weight
            if sum_agg:
                self._state.bin_sum[channel] += weight  # k(0) = 1 at the event's own sample (alpha's is 0)

        elif self.settings.kernel_type == ActivationKernelType.ALPHA:
            # For alpha kernel, event adds to auxiliary state
            self._state.alpha_aux[channel] += weight

        elif self.settings.kernel_type == ActivationKernelType.COUNT:
            self._state.activation[channel] += weight

        if max_agg:
            self._state.bin_max[channel] = max(self._state.bin_max[channel], self._state.activation[channel])

    def _track_max(self, channel: int, target_sample: int) -> None:
        """Fold the activation at samples (last update, target] into ``bin_max``, then decay to target.

        With no event in between, the activation ``k`` samples on is ``(a + u*k/tau) * exp(-k/tau)``
        (``u`` is alpha's auxiliary state, 0 for exponential), which is largest at either end of the
        interval or, for alpha, next to its peak at ``k = tau * (1 - a/u)``.
        """
        n = target_sample - self._state.samples_since_update[channel]
        if n <= 0:
            return
        tau_samples = self.settings.tau * self._state.fs
        a = self._state.activation[channel]
        u = self._state.alpha_aux[channel] if self.settings.kernel_type == ActivationKernelType.ALPHA else 0.0
        ks = [1, n]
        if u > 0:
            peak = tau_samples * (1.0 - a / u)
            ks += [math.floor(peak), math.ceil(peak)]
        ks = np.clip(ks, 1, n)
        values = (a + u * ks / tau_samples) * np.exp(-ks / tau_samples)
        self._state.bin_max[channel] = max(self._state.bin_max[channel], values.max())
        self._decay_to_sample(channel, target_sample)

    def _kernel_sum(self) -> bool:
        return (
            self.settings.aggregation == BinAggregation.SUM and self.settings.kernel_type != ActivationKernelType.COUNT
        )

    def _track_sum(self, channel: int, target_sample: int) -> None:
        """Add the activation at samples (last update, target] to ``bin_sum``, then decay to target.

        With no event in between, the activation ``k`` samples on is ``(a + u*k/tau) * r**k`` with
        ``r = exp(-1/tau)``, so the sum over ``k = 1..n`` is ``a*S0 + (u/tau)*S1`` with the geometric
        sums ``S0 = sum r**k`` and ``S1 = sum k*r**k`` in closed form.
        """
        n = target_sample - self._state.samples_since_update[channel]
        if n <= 0:
            return
        tau_samples = self.settings.tau * self._state.fs
        a = self._state.activation[channel]
        u = self._state.alpha_aux[channel] if self.settings.kernel_type == ActivationKernelType.ALPHA else 0.0
        r = math.exp(-1.0 / tau_samples)
        q = -math.expm1(-1.0 / tau_samples)  # 1 - r, without cancellation for long tau
        rn = math.exp(-n / tau_samples)
        one_minus_rn = -math.expm1(-n / tau_samples)
        s0 = r * one_minus_rn / q
        s1 = r * (one_minus_rn - n * rn * q) / (q * q)
        self._state.bin_sum[channel] += a * s0 + u * s1 / tau_samples
        self._decay_to_sample(channel, target_sample)

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

        # Boundary arithmetic is delegated to the shared schedule (samples-per-bin,
        # fractional carry, global bin index, output gain/offset).
        in_offset = message.axes["time"].offset if "time" in message.axes else 0.0
        n_carry_before = self._state.schedule.carry_count
        step = self._state.schedule.advance(n_new=n_samples, in_offset=in_offset, gain_in=1.0 / self._state.fs)
        n_bins = step.n_bins

        if n_bins == 0:
            # Not enough samples for a full bin yet (the schedule has folded these
            # samples into its carry). Still process events to update state.
            if hasattr(sparse_data, "coords") and hasattr(sparse_data, "data"):
                coords = sparse_data.coords
                values = sparse_data.data
                for event_idx in range(len(values)):
                    sample_idx = int(coords[0, event_idx])
                    channel_idx = int(coords[1, event_idx]) if coords.shape[0] > 1 else 0
                    value = float(values[event_idx])
                    self._add_event(channel_idx, sample_idx, value)
            self._state.samples_since_update -= n_samples

            # Return empty output
            return replace(
                message,
                data=np.zeros((0, n_channels), dtype=np.float64),
                axes={
                    **message.axes,
                    "time": replace(message.axes["time"], gain=step.output_gain),
                },
            )

        # Bin ends in input samples relative to *this chunk's* start. The schedule
        # returns cut points into [carry ++ new]; subtracting the pre-advance carry
        # count maps them back to chunk-local indices (identical to the legacy
        # `(spb - acc) + arange*spb` truncated formula).
        bin_ends = np.asarray(step.cut_points, dtype=np.int64) - n_carry_before

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

        elif self._kernel_sum():
            # Like MAX: `_add_event` folds samples up to each event into `bin_sum`, the rest at the bin's end.
            for bin_idx, bin_end in enumerate(bin_ends):
                bin_end_sample = int(bin_end)
                while event_idx < len(events) and events[event_idx][0] < bin_end_sample:
                    sample, channel, value = events[event_idx]
                    self._add_event(channel, sample, value)
                    event_idx += 1
                for ch in range(n_channels):
                    self._track_sum(ch, bin_end_sample - 1)
                output[bin_idx] = self._state.bin_sum
                self._state.bin_sum = np.zeros(n_channels)

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

        elif self.settings.aggregation == BinAggregation.MAX:
            # `_add_event` folds every sample up to and including each event into `bin_max`;
            # the samples after a bin's last event are folded at its end.
            for bin_idx, bin_end in enumerate(bin_ends):
                bin_end_sample = int(bin_end)
                while event_idx < len(events) and events[event_idx][0] < bin_end_sample:
                    sample, channel, value = events[event_idx]
                    self._add_event(channel, sample, value)
                    event_idx += 1
                for ch in range(n_channels):
                    self._track_max(ch, bin_end_sample - 1)
                output[bin_idx] = self._state.bin_max
                self._state.bin_max = np.full(n_channels, -np.inf)

        # Process any remaining events (for state continuity)
        while event_idx < len(events):
            sample, channel, value = events[event_idx]
            self._add_event(channel, sample, value)
            event_idx += 1

        # Update state sample counters relative to next chunk
        self._state.samples_since_update -= n_samples

        # Apply rate normalization if requested (divide by the bin's actual
        # duration to get events/second; == bin_duration in fractional mode).
        if self.settings.rate_normalize:
            output = output / step.output_gain

        return replace(
            message,
            data=output,
            axes={
                **message.axes,
                "time": AxisArray.TimeAxis(
                    fs=1.0 / step.output_gain,
                    offset=step.output_offset,
                ),
            },
        )

    def _process_dense_count_sum(self, message: AxisArray) -> AxisArray:
        """Fast path: dense input + COUNT kernel + SUM aggregation.

        Bins are summed in the input's array namespace, so accelerator-resident
        inputs (MLX, CuPy) stay on device. Uniform completed bins use direct
        reshape reductions; irregular fractional bins use cumulative sums.
        Carry-over for the partial bin spanning chunk boundaries is held in
        ``state.dense_carry`` in the active array namespace.
        """
        xp = get_namespace(message.data)
        data = message.data
        n_samples = data.shape[0]
        feature_shape = tuple(data.shape[1:])

        in_offset = message.axes["time"].offset if "time" in message.axes else 0.0
        n_carry_before = self._state.schedule.carry_count
        step = self._state.schedule.advance(n_new=n_samples, in_offset=in_offset, gain_in=1.0 / self._state.fs)
        n_bins = step.n_bins

        # Per-sample contribution: 1 per non-zero, or the value itself if scaling.
        # Use the .astype() method form so the same call works for both numpy and mlx
        # (mlx.core has no top-level astype).
        if n_samples == 0:
            contrib = xp.zeros((0,) + feature_shape, dtype=xp.float32)
        elif self.settings.scale_by_value:
            contrib = data.astype(xp.float32)
        else:
            contrib = (data != 0).astype(xp.float32)

        # Keep dense partial-bin state in the input namespace. In particular, do
        # not round-trip an MLX carry through np.asarray here: that synchronizes
        # the device on every source chunk, including chunks that close no bin.
        overflow_xp = (
            xp.zeros(feature_shape, dtype=xp.float32) if self._state.dense_carry is None else self._state.dense_carry
        )

        if n_bins == 0:
            # No complete bins this chunk — accumulate everything into the carry-over.
            new_overflow = overflow_xp + (xp.sum(contrib, axis=0) if n_samples > 0 else overflow_xp * 0)
            self._state.dense_carry = new_overflow
            return replace(
                message,
                data=xp.zeros((0,) + feature_shape, dtype=xp.float32),
                axes={
                    **message.axes,
                    "time": replace(message.axes["time"], gain=step.output_gain),
                },
            )

        # Bin boundaries in this chunk's sample space, from the shared schedule
        # (cut points into [carry ++ new] mapped back to chunk-local indices).
        bin_end_samples = np.asarray(step.cut_points, dtype=np.int64) - n_carry_before
        bin_start_samples = np.concatenate(([np.int64(0)], bin_end_samples[:-1]))

        last_bin_end = int(bin_end_samples[-1])
        bin_widths_after_first = np.diff(bin_end_samples)
        uniform_complete_bins = n_bins == 1 or np.all(bin_widths_after_first == bin_widths_after_first[0])
        # For a stream of sub-bin chunks, the occasional call that closes only
        # the carried bin is faster through the existing cumsum path. Direct
        # reduction pays off for a carry-free bin or when it can amortise setup
        # across multiple completed bins.
        use_direct_reduction = uniform_complete_bins and (n_carry_before == 0 or n_bins > 1)

        if use_direct_reduction:
            # The first bin may complete a partial bin carried from the previous
            # chunk. Every later bin is wholly inside this chunk; when those bin
            # widths are uniform, reshape + reduction avoids materialising a
            # full-size cumulative-sum array and two gather-index arrays.
            first_bin_end = int(bin_end_samples[0])
            first_bin_sum = overflow_xp + (
                xp.sum(contrib[:first_bin_end], axis=0)
                if first_bin_end > 0
                else xp.zeros(feature_shape, dtype=xp.float32)
            )
            if n_bins == 1:
                output = first_bin_sum[None, ...]
            else:
                bin_width = int(bin_widths_after_first[0])
                complete = xp.reshape(
                    contrib[first_bin_end:last_bin_end],
                    (n_bins - 1, bin_width) + feature_shape,
                )
                later_bin_sums = xp.sum(complete, axis=1)
                output = xp.concat((first_bin_sum[None, ...], later_bin_sums), axis=0)
        else:
            # Fractional schedules can produce nonuniform interior bin widths;
            # this path also handles the cheaper single carried-bin completion.
            # Cumulative sums handle arbitrary cut points without a Python loop.
            # Use cumsum (in both numpy and mlx); numpy via array_api_compat also
            # exposes `cumulative_sum`, but mlx does not.
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
        if last_bin_end < n_samples:
            new_overflow = xp.sum(contrib[last_bin_end:], axis=0)
        else:
            new_overflow = xp.zeros(feature_shape, dtype=contrib.dtype)
        self._state.dense_carry = new_overflow

        if self.settings.rate_normalize:
            output = output / step.output_gain

        return replace(
            message,
            data=output,
            axes={
                **message.axes,
                "time": AxisArray.TimeAxis(
                    fs=1.0 / step.output_gain,
                    offset=step.output_offset,
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
