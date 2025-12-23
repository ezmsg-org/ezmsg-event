import ezmsg.core as ez
import numba
import numpy as np
import numpy.typing as npt
import sparse
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


@numba.jit(nopython=True, cache=True)
def _inhomogeneous_poisson_generator(
    rates: np.ndarray,  # (n_bins,) rates for this channel
    accumulated: float,  # initial accumulated value
    threshold: float,  # initial threshold
    bin_duration: float,
    output_fs: float,
    max_events: int,
) -> tuple[np.ndarray, int, float, float]:
    """
    Inhomogeneous Poisson process event generator using the integration method.

    Returns:
        event_samples: pre-allocated array of event sample indices
        n_events: actual number of events generated
        accumulated: updated accumulated value for next chunk
        threshold: updated threshold for next chunk
    """
    event_samples = np.empty(max_events, dtype=np.int64)
    n_events = 0
    n_bins = len(rates)

    for t in range(n_bins):
        bin_start = t * bin_duration
        rate = rates[t]
        time_in_bin = 0.0

        while True:
            time_to_event = (threshold - accumulated) / rate
            event_time = time_in_bin + time_to_event

            if event_time >= bin_duration:
                # No more events in this bin
                accumulated += rate * (bin_duration - time_in_bin)
                break

            # Record event
            if n_events < max_events:
                event_sample = int((event_time + bin_start) * output_fs)
                event_samples[n_events] = event_sample
                n_events += 1

            # Update state for next event
            time_in_bin = event_time
            accumulated = 0.0
            threshold = np.random.exponential(1.0)

    return event_samples, n_events, accumulated, threshold


@numba.jit(nopython=True, parallel=True, cache=True)
def _generate_events_all_channels(
    rates_array: np.ndarray,  # (n_bins, n_channels)
    accumulated: np.ndarray,  # (n_channels,)
    threshold: np.ndarray,  # (n_channels,)
    bin_duration: float,
    output_fs: float,
    max_events_per_channel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate events for all channels in parallel.

    Returns:
        all_event_samples: (n_channels, max_events_per_channel) event sample indices
        event_counts: (n_channels,) number of events per channel
        accumulated_out: (n_channels,) updated accumulated values
        threshold_out: (n_channels,) updated thresholds
    """
    n_bins, n_channels = rates_array.shape

    # Pre-allocate output arrays
    all_event_samples = np.empty((n_channels, max_events_per_channel), dtype=np.int64)
    event_counts = np.empty(n_channels, dtype=np.int64)
    accumulated_out = np.empty(n_channels, dtype=np.float64)
    threshold_out = np.empty(n_channels, dtype=np.float64)

    # Process each channel in parallel
    for ch in numba.prange(n_channels):
        rates = rates_array[:, ch]
        samples, count, acc, thresh = _inhomogeneous_poisson_generator(
            rates,
            accumulated[ch],
            threshold[ch],
            bin_duration,
            output_fs,
            max_events_per_channel,
        )
        all_event_samples[ch, :] = samples
        event_counts[ch] = count
        accumulated_out[ch] = acc
        threshold_out[ch] = thresh

    return all_event_samples, event_counts, accumulated_out, threshold_out


@numba.jit(nopython=True, cache=True)
def _flatten_events_unsorted(
    all_event_samples: np.ndarray,  # (n_channels, max_events)
    event_counts: np.ndarray,  # (n_channels,)
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-channel event arrays into coordinate arrays (unsorted)."""
    total_events = np.sum(event_counts)

    if total_events == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    event_samples = np.empty(total_events, dtype=np.int64)
    event_channels = np.empty(total_events, dtype=np.int64)

    idx = 0
    for ch in range(len(event_counts)):
        count = event_counts[ch]
        if count > 0:
            for i in range(count):
                event_samples[idx + i] = all_event_samples[ch, i]
                event_channels[idx + i] = ch
            idx += count

    return event_samples, event_channels


class PoissonEventSettings(ez.Settings):
    output_fs: float = 30_000
    """Output sampling rate."""

    layout: str = "coo"
    """Layout of the output event train sparse array. Options are 'coo' or 'gcxs'"""

    compress_dims: list[int] | None = None
    """Dimensions to compress. Ignored if layout is 'coo'."""

    assume_counts: bool = False
    """If True, input is event counts per bin. If False, input is firing rate in Hz."""

    min_rate: float = 1e-6
    """Minimum rate to avoid division by zero."""

    max_rate: float = 500.0
    """Maximum expected firing rate (Hz). Used to pre-allocate event arrays."""


@processor_state
class PoissonEventState:
    accumulated: npt.NDArray | None = None
    """Integrated rate since last event for each channel."""

    threshold: npt.NDArray | None = None
    """Exp(1) threshold for next event for each channel."""


class PoissonEventTransformer(BaseStatefulTransformer[PoissonEventSettings, AxisArray, AxisArray, PoissonEventState]):
    def _reset_state(self, message: AxisArray) -> None:
        ch_ax = message.get_axis_idx("ch")
        n_channels = message.data.shape[ch_ax]
        self.state.accumulated = np.zeros(n_channels)
        self.state.threshold = np.random.exponential(1.0, size=n_channels)

    def _process(self, message: AxisArray) -> AxisArray:
        time_ax = message.get_axis_idx("time")
        n_bins = message.data.shape[time_ax]
        bin_duration = message.axes["time"].gain
        total_samples = n_bins * int(bin_duration * self.settings.output_fs)

        # Get rates array with shape (n_bins, n_channels), contiguous for numba
        rates_array = message.data / bin_duration if self.settings.assume_counts else message.data
        if time_ax != 0:
            rates_array = np.moveaxis(rates_array, time_ax, 0)
        rates_array = np.ascontiguousarray(np.maximum(rates_array, self.settings.min_rate))
        n_channels = rates_array.shape[1]

        # Estimate max events per channel based on actual input rates
        total_time = n_bins * bin_duration
        max_input_rate = np.max(rates_array)
        max_events_per_channel = max(int(max_input_rate * total_time * 3) + 10, 20)

        # Generate events using numba (parallel across channels)
        all_event_samples, event_counts, accumulated_out, threshold_out = _generate_events_all_channels(
            rates_array,
            self.state.accumulated,
            self.state.threshold,
            bin_duration,
            self.settings.output_fs,
            max_events_per_channel,
        )

        # Update state for next chunk
        self.state.accumulated = accumulated_out
        self.state.threshold = threshold_out

        # Flatten per-channel arrays into coordinate arrays
        event_samples, event_channels = _flatten_events_unsorted(all_event_samples, event_counts)

        # Build sparse array (COO handles sorting internally)
        if len(event_samples) > 0:
            event_samples = np.clip(event_samples, 0, total_samples - 1)
            event_coords = np.vstack([event_samples, event_channels])
            event_data = np.ones(len(event_samples), dtype=np.int8)
        else:
            event_coords = np.zeros((2, 0), dtype=np.int64)
            event_data = np.zeros(0, dtype=np.int8)

        event_array = sparse.COO(
            coords=event_coords,
            data=event_data,
            shape=(total_samples, n_channels),
        )

        if self.settings.layout == "gcxs":
            event_array = sparse.GCXS.from_coo(event_array, compressed_axes=self.settings.compress_dims)

        return replace(
            message,
            data=event_array,
            dims=["time", "ch"],
            axes={
                **message.axes,
                "time": replace(message.axes["time"], gain=1 / self.settings.output_fs),
            },
        )


class PoissonEventUnit(BaseTransformerUnit[PoissonEventSettings, AxisArray, AxisArray, PoissonEventTransformer]):
    SETTINGS = PoissonEventSettings
