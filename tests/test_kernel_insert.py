"""Unit tests for ezmsg.event.kernel_insert module."""

import numpy as np
import sparse
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.event.kernel import ArrayKernel, MultiKernel
from ezmsg.event.kernel_insert import (
    SparseKernelInserter,
    SparseKernelInserterSettings,
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
        data=np.array(values),
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


class TestSparseKernelInserterBasics:
    """Basic tests for SparseKernelInserter."""

    def test_unit_impulse_no_kernel(self):
        """Without kernel, events become unit impulses."""
        inserter = SparseKernelInserter(SparseKernelInserterSettings())

        message = make_sparse_message(
            coords=[(5, 0), (10, 1)],
            values=[1, 1],
            shape=(20, 2),
        )

        result = inserter(message)

        assert result.data.shape == (20, 2)
        assert result.data[5, 0] == 1.0
        assert result.data[10, 1] == 1.0
        assert np.sum(result.data) == 2.0

    def test_scaled_impulse(self):
        """Scale by value option."""
        inserter = SparseKernelInserter(SparseKernelInserterSettings(scale_by_value=True))

        message = make_sparse_message(
            coords=[(5, 0), (10, 0)],
            values=[3, 5],
            shape=(20, 1),
        )

        result = inserter(message)

        assert result.data[5, 0] == 3.0
        assert result.data[10, 0] == 5.0

    def test_simple_kernel_insertion(self):
        """Insert simple kernel at event location."""
        kernel_data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        inserter = SparseKernelInserter(SparseKernelInserterSettings(kernel=ArrayKernel(kernel_data)))

        message = make_sparse_message(
            coords=[(10, 0)],
            values=[1],
            shape=(20, 1),
        )

        result = inserter(message)

        # Kernel should be inserted starting at sample 10
        expected = np.zeros(20)
        expected[10:15] = kernel_data
        np.testing.assert_array_almost_equal(result.data[:, 0], expected)

    def test_overlapping_kernels_sum(self):
        """Overlapping kernels should sum additively."""
        kernel_data = np.array([1.0, 1.0, 1.0])
        inserter = SparseKernelInserter(SparseKernelInserterSettings(kernel=ArrayKernel(kernel_data)))

        # Two events 1 sample apart, kernels will overlap
        message = make_sparse_message(
            coords=[(5, 0), (6, 0)],
            values=[1, 1],
            shape=(20, 1),
        )

        result = inserter(message)

        # Sample 6 and 7 should have overlap
        assert result.data[5, 0] == 1.0  # Only first kernel
        assert result.data[6, 0] == 2.0  # Both kernels overlap
        assert result.data[7, 0] == 2.0  # Both kernels overlap
        assert result.data[8, 0] == 1.0  # Only second kernel

    def test_multiple_channels(self):
        """Handle events on different channels independently."""
        kernel_data = np.array([1.0, 2.0, 1.0])
        inserter = SparseKernelInserter(SparseKernelInserterSettings(kernel=ArrayKernel(kernel_data)))

        message = make_sparse_message(
            coords=[(5, 0), (5, 1)],
            values=[1, 1],
            shape=(20, 2),
        )

        result = inserter(message)

        # Both channels should have same kernel inserted at sample 5
        np.testing.assert_array_almost_equal(result.data[:, 0], result.data[:, 1])


class TestSparseKernelInserterChunkBoundary:
    """Test chunk boundary handling."""

    def test_kernel_extends_past_chunk(self):
        """Kernel tail carries over to next chunk."""
        kernel_data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        inserter = SparseKernelInserter(SparseKernelInserterSettings(kernel=ArrayKernel(kernel_data)))

        # Event near end of chunk - kernel will extend past
        msg1 = make_sparse_message(
            coords=[(8, 0)],
            values=[1],
            shape=(10, 1),
        )

        result1 = inserter(msg1)

        # First 2 samples of kernel should be in chunk
        assert result1.data[8, 0] == 1.0
        assert result1.data[9, 0] == 2.0

        # Second chunk with no new events
        msg2 = make_sparse_message(
            coords=[],
            values=[],
            shape=(10, 1),
            offset=0.010,  # 10 samples at 1kHz
        )

        result2 = inserter(msg2)

        # Remaining 3 samples of kernel should appear
        assert result2.data[0, 0] == 3.0
        assert result2.data[1, 0] == 2.0
        assert result2.data[2, 0] == 1.0
        assert result2.data[3, 0] == 0.0

    def test_continuity_across_chunks(self):
        """Verify seamless kernel insertion across multiple chunks."""
        kernel_data = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        inserter = SparseKernelInserter(SparseKernelInserterSettings(kernel=ArrayKernel(kernel_data)))

        # Event at end of first chunk
        msg1 = make_sparse_message(coords=[(7, 0)], values=[1], shape=(10, 1))
        result1 = inserter(msg1)

        # No events in second chunk
        msg2 = make_sparse_message(coords=[], values=[], shape=(10, 1), offset=0.010)
        result2 = inserter(msg2)

        # Concatenate results
        full_result = np.concatenate([result1.data[:, 0], result2.data[:, 0]])

        # Should have 5 consecutive 1.0 values starting at sample 7
        assert np.sum(full_result[7:12]) == 5.0
        assert full_result[6] == 0.0
        assert full_result[12] == 0.0


class TestSparseKernelInserterMultiKernel:
    """Test MultiKernel support."""

    def test_select_kernel_by_value(self):
        """Different kernels for different event values."""
        k1 = ArrayKernel(np.array([1.0, 1.0, 1.0]))
        k2 = ArrayKernel(np.array([2.0, 2.0, 2.0]))

        multi = MultiKernel({1: k1, 2: k2})
        inserter = SparseKernelInserter(SparseKernelInserterSettings(kernel=multi))

        message = make_sparse_message(
            coords=[(0, 0), (10, 0)],
            values=[1, 2],  # Different waveform IDs
            shape=(20, 1),
        )

        result = inserter(message)

        # First event gets k1 (all 1s)
        assert result.data[0, 0] == 1.0
        assert result.data[1, 0] == 1.0
        assert result.data[2, 0] == 1.0

        # Second event gets k2 (all 2s)
        assert result.data[10, 0] == 2.0
        assert result.data[11, 0] == 2.0
        assert result.data[12, 0] == 2.0


class TestSparseKernelInserterAcausal:
    """Test acausal kernel support."""

    def test_acausal_kernel(self):
        """Acausal kernels extend before event time."""
        # Symmetric kernel centered at t=0
        kernel_data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        kernel = ArrayKernel(kernel_data, pre_samples=2)

        inserter = SparseKernelInserter(SparseKernelInserterSettings(kernel=kernel))

        message = make_sparse_message(
            coords=[(5, 0)],
            values=[1],
            shape=(20, 1),
        )

        result = inserter(message)

        # Kernel should be centered at sample 5
        assert result.data[3, 0] == 1.0  # t=-2
        assert result.data[4, 0] == 2.0  # t=-1
        assert result.data[5, 0] == 3.0  # t=0
        assert result.data[6, 0] == 2.0  # t=+1
        assert result.data[7, 0] == 1.0  # t=+2


class TestSparseKernelInserterEmpty:
    """Test edge cases with empty data."""

    def test_empty_events(self):
        """Handle chunks with no events."""
        kernel_data = np.array([1.0, 2.0, 1.0])
        inserter = SparseKernelInserter(SparseKernelInserterSettings(kernel=ArrayKernel(kernel_data)))

        message = make_sparse_message(
            coords=[],
            values=[],
            shape=(20, 2),
        )

        result = inserter(message)

        assert result.data.shape == (20, 2)
        assert np.sum(result.data) == 0.0

    def test_zero_length_chunk(self):
        """Handle zero-length chunks gracefully."""
        inserter = SparseKernelInserter(SparseKernelInserterSettings())

        message = make_sparse_message(
            coords=[],
            values=[],
            shape=(0, 2),
        )

        result = inserter(message)
        assert result.data.shape == (0, 2)
