import numpy as np
import pytest
import sparse
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis, LinearAxis

from ezmsg.event.poissonevents import (
    PoissonEventSettings,
    PoissonEventTransformer,
)


def make_rate_message(
    rates: np.ndarray,
    bin_duration: float = 0.02,
    time_offset: float = 0.0,
) -> AxisArray:
    """Create an AxisArray message with firing rates.

    Args:
        rates: Array of shape (n_bins, n_channels) with firing rates in Hz.
        bin_duration: Duration of each bin in seconds.
        time_offset: Time offset for the first bin.

    Returns:
        AxisArray with the rates and appropriate axes.
    """
    n_channels = rates.shape[1]
    fs = 1.0 / bin_duration  # Sampling frequency for bins
    return AxisArray(
        data=rates,
        dims=["time", "ch"],
        axes={
            "time": LinearAxis.create_time_axis(fs=fs, offset=time_offset),
            "ch": CoordinateAxis(data=np.arange(n_channels).astype(str), dims=["ch"]),
        },
    )


class TestPoissonEventTransformer:
    def test_basic_event_generation(self):
        """Test that events are generated at approximately the expected rate."""
        np.random.seed(42)

        settings = PoissonEventSettings(output_fs=30_000)
        transformer = PoissonEventTransformer(settings)

        # 100 Hz rate, 10 bins of 20ms = 200ms total
        # Expected events: 100 * 0.2 = 20 events per channel
        n_bins = 10
        n_channels = 4
        rate = 100.0
        bin_duration = 0.02

        rates = np.full((n_bins, n_channels), rate)
        msg = make_rate_message(rates, bin_duration=bin_duration)

        result = transformer(msg)

        assert isinstance(result.data, sparse.COO)
        total_events = result.data.nnz

        # With 4 channels at 100 Hz for 200ms, expect ~80 events total
        # Allow reasonable variance (Poisson process)
        expected_total = rate * bin_duration * n_bins * n_channels
        assert total_events > expected_total * 0.5
        assert total_events < expected_total * 1.5

    def test_output_shape_and_sampling_rate(self):
        """Test that output has correct shape and time axis gain."""
        settings = PoissonEventSettings(output_fs=30_000)
        transformer = PoissonEventTransformer(settings)

        n_bins = 5
        n_channels = 8
        bin_duration = 0.02

        rates = np.full((n_bins, n_channels), 50.0)
        msg = make_rate_message(rates, bin_duration=bin_duration)

        result = transformer(msg)

        # Expected samples: n_bins * bin_duration * output_fs
        expected_samples = int(n_bins * bin_duration * 30_000)
        assert result.data.shape == (expected_samples, n_channels)
        assert result.axes["time"].gain == pytest.approx(1 / 30_000)

    def test_low_rate_generates_events_over_time(self):
        """Test that low rates eventually generate events across multiple chunks."""
        np.random.seed(123)

        settings = PoissonEventSettings(output_fs=30_000)
        transformer = PoissonEventTransformer(settings)

        # 10 Hz rate with 20ms bins: expected 0.2 events per bin per channel
        # Over 50 bins (1 second), expect ~10 events per channel
        n_bins = 50
        n_channels = 2
        rate = 10.0
        bin_duration = 0.02

        rates = np.full((n_bins, n_channels), rate)
        msg = make_rate_message(rates, bin_duration=bin_duration)

        result = transformer(msg)

        # Should have some events despite low rate
        assert result.data.nnz > 0

        # Check approximately correct rate
        total_time = n_bins * bin_duration
        expected_events = rate * total_time * n_channels
        assert result.data.nnz > expected_events * 0.3
        assert result.data.nnz < expected_events * 2.0

    def test_state_persistence_across_chunks(self):
        """Test that state carries over correctly between processing calls."""
        np.random.seed(456)

        settings = PoissonEventSettings(output_fs=30_000)
        transformer = PoissonEventTransformer(settings)

        n_channels = 4
        rate = 50.0
        bin_duration = 0.02
        n_bins_per_chunk = 5

        total_events = 0
        for chunk_idx in range(10):
            rates = np.full((n_bins_per_chunk, n_channels), rate)
            time_offset = chunk_idx * n_bins_per_chunk * bin_duration
            msg = make_rate_message(rates, bin_duration=bin_duration, time_offset=time_offset)

            result = transformer(msg)
            total_events += result.data.nnz

        # Total time: 10 chunks * 5 bins * 20ms = 1 second
        # Expected: 50 Hz * 1s * 4 channels = 200 events
        expected_total = 200
        assert total_events > expected_total * 0.6
        assert total_events < expected_total * 1.5

    def test_rate_change_low_to_high(self):
        """Test the key scenario: low rate followed by high rate."""
        np.random.seed(789)

        settings = PoissonEventSettings(output_fs=30_000)
        transformer = PoissonEventTransformer(settings)

        n_channels = 10
        bin_duration = 0.02

        # First chunk: very low rate (should accumulate progress toward events)
        low_rates = np.full((5, n_channels), 1.0)  # 1 Hz
        msg1 = make_rate_message(low_rates, bin_duration=bin_duration, time_offset=0.0)
        result1 = transformer(msg1)

        # Second chunk: high rate (should trigger events quickly)
        high_rates = np.full((5, n_channels), 100.0)  # 100 Hz
        msg2 = make_rate_message(high_rates, bin_duration=bin_duration, time_offset=0.1)
        result2 = transformer(msg2)

        # The high-rate chunk should have many events
        # Expected for 100 Hz * 0.1s * 10 channels = 100 events
        assert result2.data.nnz > 50  # At least half expected

        # The low-rate chunk might have few or no events
        # 1 Hz * 0.1s * 10 channels = 1 event expected
        assert result1.data.nnz < 20  # Should be very few

    def test_rate_change_high_to_low(self):
        """Test high rate followed by low rate - accumulated state should persist."""
        np.random.seed(101)

        settings = PoissonEventSettings(output_fs=30_000)
        transformer = PoissonEventTransformer(settings)

        n_channels = 8
        bin_duration = 0.02

        # First chunk: high rate
        high_rates = np.full((5, n_channels), 100.0)
        msg1 = make_rate_message(high_rates, bin_duration=bin_duration, time_offset=0.0)
        result1 = transformer(msg1)

        # Store the state after high-rate processing
        _ = transformer.state.accumulated.copy()

        # Second chunk: very low rate
        low_rates = np.full((50, n_channels), 0.1)  # 0.1 Hz - very low
        msg2 = make_rate_message(low_rates, bin_duration=bin_duration, time_offset=0.1)
        result2 = transformer(msg2)

        # High-rate chunk should have many events
        assert result1.data.nnz > 20

        # Low-rate chunk: some channels might event due to accumulated progress
        # from the high-rate chunk, but overall should be sparse
        # 0.1 Hz * 1s * 8 channels = 0.8 events expected from rate alone
        # But accumulated state might cause a few more
        assert result2.data.nnz < 30

    def test_zero_rate_no_events(self):
        """Test that zero rate produces no events."""
        settings = PoissonEventSettings(output_fs=30_000, min_rate=1e-10)
        transformer = PoissonEventTransformer(settings)

        n_bins = 10
        n_channels = 4
        bin_duration = 0.02

        # Use min_rate (effectively zero)
        rates = np.full((n_bins, n_channels), 1e-10)
        msg = make_rate_message(rates, bin_duration=bin_duration)

        result = transformer(msg)

        # Should have no events (or extremely few due to numerical precision)
        assert result.data.nnz == 0

    def test_gcxs_layout(self):
        """Test that GCXS layout option works correctly."""
        np.random.seed(202)

        settings = PoissonEventSettings(
            output_fs=30_000,
            layout="gcxs",
            compress_dims=[0],
        )
        transformer = PoissonEventTransformer(settings)

        rates = np.full((5, 4), 50.0)
        msg = make_rate_message(rates, bin_duration=0.02)

        result = transformer(msg)

        assert isinstance(result.data, sparse.GCXS)

    def test_assume_counts_mode(self):
        """Test that assume_counts correctly interprets input as event counts."""
        np.random.seed(303)

        settings = PoissonEventSettings(output_fs=30_000, assume_counts=True)
        transformer = PoissonEventTransformer(settings)

        n_bins = 10
        n_channels = 4
        bin_duration = 0.02

        # Input as counts: 2 events per bin = 100 Hz rate
        counts = np.full((n_bins, n_channels), 2.0)
        msg = make_rate_message(counts, bin_duration=bin_duration)

        result = transformer(msg)

        # Expected rate = 2 / 0.02 = 100 Hz
        # Expected events = 100 * 0.2 * 4 = 80
        expected = 80
        assert result.data.nnz > expected * 0.5
        assert result.data.nnz < expected * 1.5

    def test_event_times_within_bounds(self):
        """Test that all event times fall within valid sample range."""
        np.random.seed(404)

        settings = PoissonEventSettings(output_fs=30_000)
        transformer = PoissonEventTransformer(settings)

        n_bins = 5
        n_channels = 8
        bin_duration = 0.02

        rates = np.full((n_bins, n_channels), 200.0)  # High rate
        msg = make_rate_message(rates, bin_duration=bin_duration)

        result = transformer(msg)

        total_samples = int(n_bins * bin_duration * 30_000)

        # All time coordinates should be within bounds
        if result.data.nnz > 0:
            time_coords = result.data.coords[0]
            assert np.all(time_coords >= 0)
            assert np.all(time_coords < total_samples)

            ch_coords = result.data.coords[1]
            assert np.all(ch_coords >= 0)
            assert np.all(ch_coords < n_channels)

    def test_channel_axis_position(self):
        """Test that channel axis in non-standard position is handled correctly."""
        np.random.seed(505)

        settings = PoissonEventSettings(output_fs=30_000)
        transformer = PoissonEventTransformer(settings)

        n_bins = 5
        n_channels = 4
        bin_duration = 0.02
        fs = 1.0 / bin_duration

        # Create message with channel axis first (transposed)
        rates = np.full((n_channels, n_bins), 50.0)
        msg = AxisArray(
            data=rates,
            dims=["ch", "time"],
            axes={
                "time": LinearAxis.create_time_axis(fs=fs, offset=0.0),
                "ch": CoordinateAxis(data=np.arange(n_channels).astype(str), dims=["ch"]),
            },
        )

        result = transformer(msg)

        expected_samples = int(n_bins * bin_duration * 30_000)
        assert result.data.shape == (expected_samples, n_channels)

    def test_statistical_properties(self):
        """Test that event intervals follow expected exponential distribution."""
        np.random.seed(606)

        settings = PoissonEventSettings(output_fs=30_000)
        transformer = PoissonEventTransformer(settings)

        # Long duration to get good statistics
        n_bins = 100
        n_channels = 1
        rate = 50.0
        bin_duration = 0.02

        rates = np.full((n_bins, n_channels), rate)
        msg = make_rate_message(rates, bin_duration=bin_duration)

        result = transformer(msg)

        # Extract event times for the single channel
        event_samples = result.data.coords[0]
        event_times = event_samples / settings.output_fs

        if len(event_times) > 10:
            # Calculate inter-event intervals
            sorted_times = np.sort(event_times)
            isis = np.diff(sorted_times)

            # Mean ISI should be approximately 1/rate
            expected_mean_isi = 1 / rate
            actual_mean_isi = np.mean(isis)

            # Allow 50% tolerance due to finite sample size
            assert actual_mean_isi > expected_mean_isi * 0.5
            assert actual_mean_isi < expected_mean_isi * 2.0
