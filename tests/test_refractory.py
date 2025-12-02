import numpy as np
import pytest
import sparse
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.event.refractory import RefractoryTransformer


class TestRefractoryTransformer:
    """Test suite for RefractoryTransformer."""

    def test_no_refractory_passthrough(self):
        """When refractory duration is 0, events should pass through unchanged."""
        fs = 1000.0
        n_samples = 100
        n_chans = 4

        # Create sparse data with events
        coords = [[0, 1, 2, 3], [10, 20, 30, 40]]  # ch, time
        data = np.array([True, True, True, True])
        sparse_data = sparse.COO(coords, data, shape=(n_chans, n_samples))

        msg = AxisArray(
            data=sparse_data,
            dims=["ch", "time"],
            axes={"time": AxisArray.Axis.TimeAxis(fs=fs, offset=0.0)},
            key="test_passthrough",
        )

        transformer = RefractoryTransformer(dur=0.0)
        result = transformer(msg)

        assert result.data.nnz == 4  # All events should pass through

    def test_refractory_filters_close_events(self):
        """Events within refractory period should be filtered out."""
        fs = 1000.0
        n_samples = 100
        n_chans = 1
        refrac_dur = 0.010  # 10ms = 10 samples at 1kHz

        # Create sparse data with events: samples 10, 15, 30, 35
        # Events at 10 and 30 should pass, events at 15 and 35 should be filtered
        coords = [[0, 0, 0, 0], [10, 15, 30, 35]]  # ch, time
        data = np.array([True, True, True, True])
        sparse_data = sparse.COO(coords, data, shape=(n_chans, n_samples))

        msg = AxisArray(
            data=sparse_data,
            dims=["ch", "time"],
            axes={"time": AxisArray.Axis.TimeAxis(fs=fs, offset=0.0)},
            key="test_filter",
        )

        transformer = RefractoryTransformer(dur=refrac_dur)
        result = transformer(msg)

        # Events at 15 and 35 are within 10 samples of 10 and 30, so should be filtered
        assert result.data.nnz == 2
        result_times = result.data.coords[1]
        assert 10 in result_times
        assert 30 in result_times
        assert 15 not in result_times
        assert 35 not in result_times

    def test_refractory_independent_channels(self):
        """Refractory period should be enforced independently per channel."""
        fs = 1000.0
        n_samples = 50
        n_chans = 2
        refrac_dur = 0.010  # 10ms = 10 samples

        # Channel 0: events at 10, 15 (15 should be filtered)
        # Channel 1: events at 10, 15 (15 should be filtered)
        coords = [[0, 0, 1, 1], [10, 15, 10, 15]]
        data = np.array([True, True, True, True])
        sparse_data = sparse.COO(coords, data, shape=(n_chans, n_samples))

        msg = AxisArray(
            data=sparse_data,
            dims=["ch", "time"],
            axes={"time": AxisArray.Axis.TimeAxis(fs=fs, offset=0.0)},
            key="test_channels",
        )

        transformer = RefractoryTransformer(dur=refrac_dur)
        result = transformer(msg)

        # Each channel should have only one event (at sample 10)
        assert result.data.nnz == 2
        # Both channels should have an event at sample 10
        ch_inds = result.data.coords[0]
        time_inds = result.data.coords[1]
        assert np.all(time_inds == 10)
        assert set(ch_inds) == {0, 1}

    def test_refractory_across_chunks(self):
        """Refractory period should be enforced across message boundaries."""
        fs = 1000.0
        n_samples = 50
        n_chans = 1
        refrac_dur = 0.020  # 20ms = 20 samples

        transformer = RefractoryTransformer(dur=refrac_dur)

        # First chunk: event at sample 45
        coords1 = [[0], [45]]
        data1 = np.array([True])
        sparse_data1 = sparse.COO(coords1, data1, shape=(n_chans, n_samples))
        msg1 = AxisArray(
            data=sparse_data1,
            dims=["ch", "time"],
            axes={"time": AxisArray.Axis.TimeAxis(fs=fs, offset=0.0)},
            key="test_across_chunks",
        )

        # Second chunk: event at sample 5 (which is 5 samples after chunk boundary)
        # Total elapsed since last event would be 5 + (50-45) = 10 samples < 20 refrac
        coords2 = [[0], [5]]
        data2 = np.array([True])
        sparse_data2 = sparse.COO(coords2, data2, shape=(n_chans, n_samples))
        msg2 = AxisArray(
            data=sparse_data2,
            dims=["ch", "time"],
            axes={"time": AxisArray.Axis.TimeAxis(fs=fs, offset=0.050)},
            key="test_across_chunks",
        )

        result1 = transformer(msg1)
        result2 = transformer(msg2)

        # First chunk event should pass
        assert result1.data.nnz == 1
        # Second chunk event should be filtered (only 10 samples since last event)
        assert result2.data.nnz == 0

    def test_empty_sparse_message(self):
        """Empty sparse messages should pass through without errors."""
        fs = 1000.0
        n_samples = 100
        n_chans = 4
        refrac_dur = 0.010

        # Create empty sparse data
        coords = [[], []]
        data = np.array([], dtype=bool)
        sparse_data = sparse.COO(coords, data, shape=(n_chans, n_samples))

        msg = AxisArray(
            data=sparse_data,
            dims=["ch", "time"],
            axes={"time": AxisArray.Axis.TimeAxis(fs=fs, offset=0.0)},
            key="test_empty",
        )

        transformer = RefractoryTransformer(dur=refrac_dur)
        result = transformer(msg)

        assert result.data.nnz == 0

    def test_cascade_filtering(self):
        """Test that filtering cascades correctly when multiple events are close."""
        fs = 1000.0
        n_samples = 100
        n_chans = 1
        refrac_dur = 0.010  # 10 samples

        # Events at 10, 15, 18, 30
        # 10 passes, 15 filtered (5 from 10), 18 should be checked against 10 (not 15)
        # 18 is 8 from 10, so filtered. 30 passes (20 from 10)
        coords = [[0, 0, 0, 0], [10, 15, 18, 30]]
        data = np.array([True, True, True, True])
        sparse_data = sparse.COO(coords, data, shape=(n_chans, n_samples))

        msg = AxisArray(
            data=sparse_data,
            dims=["ch", "time"],
            axes={"time": AxisArray.Axis.TimeAxis(fs=fs, offset=0.0)},
            key="test_cascade",
        )

        transformer = RefractoryTransformer(dur=refrac_dur)
        result = transformer(msg)

        assert result.data.nnz == 2
        result_times = result.data.coords[1]
        assert 10 in result_times
        assert 30 in result_times

    @pytest.mark.parametrize("time_axis_position", [0, 1])
    def test_different_time_axis_positions(self, time_axis_position: int):
        """Test that the transformer works with time axis in different positions."""
        fs = 1000.0
        n_samples = 100
        n_chans = 4
        refrac_dur = 0.010

        if time_axis_position == 0:
            # time, ch
            coords = [[10, 20], [0, 1]]
            shape = (n_samples, n_chans)
            dims = ["time", "ch"]
        else:
            # ch, time
            coords = [[0, 1], [10, 20]]
            shape = (n_chans, n_samples)
            dims = ["ch", "time"]

        data = np.array([True, True])
        sparse_data = sparse.COO(coords, data, shape=shape)

        msg = AxisArray(
            data=sparse_data,
            dims=dims,
            axes={"time": AxisArray.Axis.TimeAxis(fs=fs, offset=0.0)},
            key="test_time_axis",
        )

        transformer = RefractoryTransformer(dur=refrac_dur)
        result = transformer(msg)

        # Both events are far enough apart to pass
        assert result.data.nnz == 2
