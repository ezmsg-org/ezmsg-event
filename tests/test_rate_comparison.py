"""Compare Rate and event_rate outputs to verify equivalence."""

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
from ezmsg.event.rate import EventRateSettings, Rate


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
        data=np.array(values, dtype=int),
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


class TestRateComparison:
    """Compare Rate and event_rate outputs."""

    def test_single_channel_identical_values(self):
        """Rate and event_rate should produce identical values for single channel."""
        bin_duration = 0.010  # 10ms bins at 1kHz = 10 samples/bin
        fs = 1000.0
        n_samples = 100  # 10 bins

        # Create Rate processor
        rate_proc = Rate(EventRateSettings(bin_duration=bin_duration))

        # Create event_rate processor
        event_rate_proc = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.COUNT,
                bin_duration=bin_duration,
                aggregation=BinAggregation.SUM,
                normalize=False,
                rate_normalize=True,
            )
        )

        # Create test message with events at known positions
        # 3 events in bin 0 (samples 0-9), 2 in bin 1 (10-19), 0 in bin 2, 1 in bin 3
        message = make_sparse_message(
            coords=[(2, 0), (5, 0), (8, 0), (12, 0), (17, 0), (35, 0)],
            values=[1, 1, 1, 1, 1, 1],
            shape=(n_samples, 1),
            fs=fs,
        )

        rate_result = rate_proc(message)
        event_rate_result = event_rate_proc(message)

        # Both should have same number of bins
        assert rate_result.data.shape[0] == event_rate_result.data.shape[0]

        # Values should be identical (both are events/second)
        np.testing.assert_array_almost_equal(
            rate_result.data,
            event_rate_result.data,
            decimal=10,
            err_msg="Rate and event_rate produced different values",
        )

    def test_multi_channel_identical_values(self):
        """Rate and event_rate should produce identical values for multiple channels."""
        bin_duration = 0.020  # 20ms bins
        fs = 1000.0
        n_samples = 100  # 5 bins

        rate_proc = Rate(EventRateSettings(bin_duration=bin_duration))
        event_rate_proc = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.COUNT,
                bin_duration=bin_duration,
                aggregation=BinAggregation.SUM,
                normalize=False,
                rate_normalize=True,
            )
        )

        # Events across 3 channels
        message = make_sparse_message(
            coords=[
                (5, 0),
                (15, 0),  # Channel 0: 1 event in bin 0, 1 in bin 1
                (8, 1),
                (12, 1),
                (25, 1),  # Channel 1: 1 in bin 0, 1 in bin 1, 1 in bin 2
                (45, 2),  # Channel 2: 1 in bin 2
            ],
            values=[1, 1, 1, 1, 1, 1],
            shape=(n_samples, 3),
            fs=fs,
        )

        rate_result = rate_proc(message)
        event_rate_result = event_rate_proc(message)

        assert rate_result.data.shape == event_rate_result.data.shape

        np.testing.assert_array_almost_equal(
            rate_result.data,
            event_rate_result.data,
            decimal=10,
        )

    def test_empty_events(self):
        """Both should handle empty events identically."""
        bin_duration = 0.010
        fs = 1000.0
        n_samples = 50

        rate_proc = Rate(EventRateSettings(bin_duration=bin_duration))
        event_rate_proc = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.COUNT,
                bin_duration=bin_duration,
                aggregation=BinAggregation.SUM,
                normalize=False,
                rate_normalize=True,
            )
        )

        message = make_sparse_message(
            coords=[],
            values=[],
            shape=(n_samples, 2),
            fs=fs,
        )

        rate_result = rate_proc(message)
        event_rate_result = event_rate_proc(message)

        assert rate_result.data.shape == event_rate_result.data.shape
        np.testing.assert_array_almost_equal(
            rate_result.data,
            event_rate_result.data,
            decimal=10,
        )

    def test_streaming_chunks(self):
        """Both should handle streaming chunks identically."""
        bin_duration = 0.015  # 15ms bins = 15 samples at 1kHz
        fs = 1000.0

        rate_proc = Rate(EventRateSettings(bin_duration=bin_duration))
        event_rate_proc = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.COUNT,
                bin_duration=bin_duration,
                aggregation=BinAggregation.SUM,
                normalize=False,
                rate_normalize=True,
            )
        )

        # First chunk: 20 samples (1 full bin + 5 samples overflow)
        msg1 = make_sparse_message(
            coords=[(5, 0), (12, 0)],
            values=[1, 1],
            shape=(20, 1),
            fs=fs,
            offset=0.0,
        )

        rate_result1 = rate_proc(msg1)
        event_rate_result1 = event_rate_proc(msg1)

        # Second chunk: 20 samples
        msg2 = make_sparse_message(
            coords=[(3, 0), (18, 0)],
            values=[1, 1],
            shape=(20, 1),
            fs=fs,
            offset=0.020,
        )

        rate_result2 = rate_proc(msg2)
        event_rate_result2 = event_rate_proc(msg2)

        # Compare shapes
        assert rate_result1.data.shape == event_rate_result1.data.shape
        assert rate_result2.data.shape == event_rate_result2.data.shape

        # Compare values
        np.testing.assert_array_almost_equal(
            rate_result1.data,
            event_rate_result1.data,
            decimal=10,
        )
        np.testing.assert_array_almost_equal(
            rate_result2.data,
            event_rate_result2.data,
            decimal=10,
        )

    def test_output_rate_metadata(self):
        """Both should produce correct output sample rate metadata."""
        bin_duration = 0.025  # 40 Hz output
        fs = 1000.0
        n_samples = 100

        rate_proc = Rate(EventRateSettings(bin_duration=bin_duration))
        event_rate_proc = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.COUNT,
                bin_duration=bin_duration,
                aggregation=BinAggregation.SUM,
                normalize=False,
                rate_normalize=True,
            )
        )

        message = make_sparse_message(
            coords=[(10, 0)],
            values=[1],
            shape=(n_samples, 1),
            fs=fs,
        )

        rate_result = rate_proc(message)
        event_rate_result = event_rate_proc(message)

        # Both should have output fs = 1/bin_duration = 40 Hz
        rate_output_fs = 1.0 / rate_result.axes["time"].gain
        event_rate_output_fs = 1.0 / event_rate_result.axes["time"].gain

        assert rate_output_fs == pytest.approx(40.0)
        assert event_rate_output_fs == pytest.approx(40.0)

    def test_rate_values_correct(self):
        """Verify the actual rate values are correct (events/second)."""
        bin_duration = 0.010  # 10ms bins
        fs = 1000.0

        event_rate_proc = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.COUNT,
                bin_duration=bin_duration,
                aggregation=BinAggregation.SUM,
                normalize=False,
                rate_normalize=True,
            )
        )

        # 3 events in one 10ms bin = 300 events/second
        message = make_sparse_message(
            coords=[(2, 0), (5, 0), (8, 0)],
            values=[1, 1, 1],
            shape=(10, 1),
            fs=fs,
        )

        result = event_rate_proc(message)

        # 3 events / 0.010 seconds = 300 events/second
        assert result.data[0, 0] == pytest.approx(300.0)
