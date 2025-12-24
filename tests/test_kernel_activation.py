"""Unit tests for ezmsg.event.kernel_activation module."""

import numpy as np
import pytest
import sparse
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.event.kernel_activation import (
    ActivationKernelType,
    BinAggregation,
    BinnedKernelActivation,
    BinnedKernelActivationSettings,
)


def make_sparse_message(
    coords: list[tuple[int, int]],
    values: list[int | float],
    shape: tuple[int, int],
    fs: float = 1000.0,
    offset: float = 0.0,
) -> AxisArray:
    """Create a sparse AxisArray message for testing."""
    if coords:
        coords_array = np.array(coords).T
    else:
        coords_array = np.array([[], []], dtype=int)

    data = sparse.COO(
        coords=coords_array,
        data=np.array(values, dtype=float),
        shape=shape,
    )

    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=offset),
            "ch": AxisArray.CoordinateAxis(
                data=np.array([f"Ch{i}" for i in range(shape[1])]),
                dims=["ch"],
            ),
        },
    )


class TestEventCount:
    """Tests for simple event counting."""

    def test_basic_count(self):
        """Count events per bin."""
        counter = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.COUNT,
                bin_duration=0.010,
                aggregation=BinAggregation.SUM,
                normalize=False,
            )
        )

        # 2 events in first bin, 1 in second
        message = make_sparse_message(
            coords=[(2, 0), (5, 0), (15, 0)],
            values=[1, 1, 1],
            shape=(30, 1),  # 30 samples = 3 bins
            fs=1000.0,
        )

        result = counter(message)

        assert result.data.shape[0] == 3  # 3 bins
        assert result.data[0, 0] == 2.0  # 2 events in first bin
        assert result.data[1, 0] == 1.0  # 1 event in second bin
        assert result.data[2, 0] == 0.0  # 0 events in third bin

    def test_weighted_count(self):
        """Count with event values as weights."""
        counter = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.COUNT,
                bin_duration=0.010,
                aggregation=BinAggregation.SUM,
                normalize=False,
                scale_by_value=True,
            )
        )

        message = make_sparse_message(
            coords=[(2, 0), (5, 0)],
            values=[3, 5],  # Different weights
            shape=(10, 1),
            fs=1000.0,
        )

        result = counter(message)

        assert result.data[0, 0] == 8.0  # 3 + 5

    def test_multi_channel_count(self):
        """Count independently per channel."""
        counter = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.COUNT,
                bin_duration=0.010,
                aggregation=BinAggregation.SUM,
                normalize=False,
            )
        )

        message = make_sparse_message(
            coords=[(2, 0), (3, 1), (5, 0), (7, 1)],
            values=[1, 1, 1, 1],
            shape=(10, 2),
            fs=1000.0,
        )

        result = counter(message)

        assert result.data[0, 0] == 2.0  # Channel 0: 2 events
        assert result.data[0, 1] == 2.0  # Channel 1: 2 events


class TestExponentialActivation:
    """Tests for exponential kernel activation."""

    def test_single_event_decay(self):
        """Single event should decay exponentially."""
        # tau = 50ms, bins = 10ms, fs = 1000 Hz
        activation = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.EXPONENTIAL,
                tau=0.050,
                bin_duration=0.010,
                aggregation=BinAggregation.LAST,
                normalize=False,  # Easier to verify
            )
        )

        # Single event at sample 0
        message = make_sparse_message(
            coords=[(0, 0)],
            values=[1],
            shape=(50, 1),  # 50ms = 5 bins
            fs=1000.0,
        )

        result = activation(message)

        # Activation should decay exponentially
        # At bin ends: 10, 20, 30, 40, 50 ms
        tau_samples = 50  # 50ms * 1000 Hz
        bin_samples = 10

        expected = []
        for i in range(5):
            bin_end = (i + 1) * bin_samples
            expected.append(np.exp(-bin_end / tau_samples))

        for i in range(5):
            assert result.data[i, 0] == pytest.approx(expected[i], rel=0.01)

    def test_multiple_events_sum(self):
        """Multiple events should sum in activation."""
        activation = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.EXPONENTIAL,
                tau=0.100,  # 100ms
                bin_duration=0.050,  # 50ms bins
                aggregation=BinAggregation.LAST,
                normalize=False,
            )
        )

        # Two events at consecutive samples (sparse.COO dedupes same coords)
        message = make_sparse_message(
            coords=[(0, 0), (1, 0)],
            values=[1, 1],
            shape=(50, 1),
            fs=1000.0,
        )

        result = activation(message)

        # Activation should be approximately 2x a single event
        # Event at t=0 decays for 50 samples, event at t=1 decays for 49 samples
        tau_samples = 100
        expected = np.exp(-50 / tau_samples) + np.exp(-49 / tau_samples)
        assert result.data[0, 0] == pytest.approx(expected, rel=0.01)

    def test_output_rate(self):
        """Output should have correct sample rate."""
        activation = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.EXPONENTIAL,
                tau=0.050,
                bin_duration=0.020,  # 50 Hz output
            )
        )

        message = make_sparse_message(
            coords=[(0, 0)],
            values=[1],
            shape=(100, 1),  # 100ms
            fs=1000.0,
        )

        result = activation(message)

        # 100ms / 20ms bins = 5 bins
        assert result.data.shape[0] == 5

        # Output fs should be 1/0.020 = 50 Hz (LinearAxis uses gain = 1/fs)
        output_fs = 1.0 / result.axes["time"].gain
        assert output_fs == pytest.approx(50.0)


class TestAlphaActivation:
    """Tests for alpha kernel activation."""

    def test_alpha_rises_then_decays(self):
        """Alpha kernel should rise then decay."""
        activation = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.ALPHA,
                tau=0.020,  # 20ms peak time
                bin_duration=0.010,  # 10ms bins
                aggregation=BinAggregation.LAST,
                normalize=False,
            )
        )

        # Single event at sample 0
        message = make_sparse_message(
            coords=[(0, 0)],
            values=[1],
            shape=(100, 1),  # 100ms
            fs=1000.0,
        )

        result = activation(message)

        # Alpha kernel peaks at t = tau
        # With 10ms bins, peak should be around bin 2 (20ms)
        values = result.data[:, 0]

        # Should rise initially
        assert values[1] > values[0]

        # Should decay after peak
        peak_bin = np.argmax(values)
        assert values[peak_bin + 1] < values[peak_bin]


class TestBinAccumulation:
    """Test proper bin accumulation across chunks."""

    def test_partial_bin_carries_over(self):
        """Partial bins should accumulate across chunks."""
        counter = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.COUNT,
                bin_duration=0.015,
                aggregation=BinAggregation.SUM,
                normalize=False,
            )
        )

        # First chunk: 10ms (not enough for a bin)
        msg1 = make_sparse_message(
            coords=[(5, 0)],
            values=[1],
            shape=(10, 1),  # 10ms
            fs=1000.0,
        )

        result1 = counter(msg1)
        assert result1.data.shape[0] == 0  # No complete bins

        # Second chunk: 10ms more (total 20ms = 1 bin with 5ms overflow)
        msg2 = make_sparse_message(
            coords=[(3, 0)],
            values=[1],
            shape=(10, 1),
            fs=1000.0,
            offset=0.010,
        )

        result2 = counter(msg2)
        assert result2.data.shape[0] == 1  # 1 complete bin
        assert result2.data[0, 0] == 2.0  # Both events in first bin


class TestChunkContinuity:
    """Test state continuity across chunks."""

    def test_exponential_continuity(self):
        """Exponential activation should be continuous across chunks."""
        activation = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.EXPONENTIAL,
                tau=0.100,  # 100ms
                bin_duration=0.010,  # 10ms bins
                normalize=False,
            )
        )

        # Event in first chunk
        msg1 = make_sparse_message(
            coords=[(0, 0)],
            values=[1],
            shape=(20, 1),  # 20ms = 2 bins
            fs=1000.0,
        )

        result1 = activation(msg1)

        # No events in second chunk - should continue decaying
        msg2 = make_sparse_message(
            coords=[],
            values=[],
            shape=(20, 1),
            fs=1000.0,
            offset=0.020,
        )

        result2 = activation(msg2)

        # Concatenate results
        all_values = np.concatenate([result1.data[:, 0], result2.data[:, 0]])

        # All values should be monotonically decreasing (exponential decay)
        for i in range(1, len(all_values)):
            assert all_values[i] < all_values[i - 1]


class TestEmptyInput:
    """Test handling of empty inputs."""

    def test_empty_events(self):
        """Handle chunks with no events."""
        activation = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.EXPONENTIAL,
                tau=0.050,
                bin_duration=0.010,
            )
        )

        message = make_sparse_message(
            coords=[],
            values=[],
            shape=(30, 2),
            fs=1000.0,
        )

        result = activation(message)

        assert result.data.shape == (3, 2)  # 3 bins
        assert np.all(result.data == 0.0)

    def test_insufficient_samples_for_bin(self):
        """Handle chunks smaller than one bin."""
        activation = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.EXPONENTIAL,
                tau=0.050,
                bin_duration=0.020,
            )
        )

        # Only 10 samples at 1kHz = 10ms < 20ms bin
        message = make_sparse_message(
            coords=[(5, 0)],
            values=[1],
            shape=(10, 1),
            fs=1000.0,
        )

        result = activation(message)

        # Should return empty output (no complete bins)
        assert result.data.shape[0] == 0


class TestAggregationModes:
    """Test different bin aggregation modes."""

    def test_sum_aggregation(self):
        """SUM aggregation for counting."""
        activation = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.COUNT,
                bin_duration=0.010,
                aggregation=BinAggregation.SUM,
            )
        )

        message = make_sparse_message(
            coords=[(2, 0), (5, 0), (7, 0)],
            values=[1, 1, 1],
            shape=(10, 1),
            fs=1000.0,
        )

        result = activation(message)

        assert result.data[0, 0] == 3.0  # Sum of all events in bin
