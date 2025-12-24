"""
Insert kernels at sparse event locations to produce dense signals.

This module provides efficient sparse-to-dense conversion by inserting
kernel waveforms at event locations. Overlapping kernels are summed.
"""

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from ezmsg.baseproc import BaseStatefulTransformer, BaseTransformerUnit, processor_state
from ezmsg.util.messages.axisarray import AxisArray, replace

from .kernel import Kernel, MultiKernel


class SparseKernelInserterSettings(ez.Settings):
    """Settings for SparseKernelInserter."""

    kernel: Kernel | MultiKernel | None = None
    """
    Kernel to insert at event locations.
    - Kernel: Same kernel for all events.
    - MultiKernel: Different kernels based on event value.
    - None: Events are treated as unit impulses (delta functions).
    """

    scale_by_value: bool = False
    """
    If True, scale kernel amplitude by event value.
    If False, event value is used only for MultiKernel selection.
    """

    output_dtype: npt.DTypeLike = np.float64
    """Data type for output array."""


@processor_state
class SparseKernelInserterState:
    """State for SparseKernelInserter."""

    # Pending contributions that overlap into next chunk
    # Shape: (pending_samples, n_channels)
    pending: npt.NDArray[np.floating] | None = None

    # Number of pending samples (may be less than pending.shape[0] if reused)
    pending_length: int = 0


class SparseKernelInserter(
    BaseStatefulTransformer[
        SparseKernelInserterSettings,
        AxisArray,
        AxisArray,
        SparseKernelInserterState,
    ]
):
    """
    Insert kernels at sparse event locations, producing dense output.

    Input: AxisArray with sparse.COO data where:
        - coords[0]: sample indices (time)
        - coords[1]: channel indices
        - data: event values (used for MultiKernel selection or scaling)

    Output: AxisArray with dense data containing inserted kernels.

    Features:
        - Handles chunk boundaries seamlessly (kernel tails carry over)
        - Overlapping kernels are summed additively
        - Supports acausal kernels (pre_samples > 0)
        - Efficient O(n_events * kernel_length) instead of dense convolution
    """

    def _get_max_kernel_length(self) -> int:
        """Get maximum kernel length for buffer allocation."""
        kernel = self.settings.kernel
        if kernel is None:
            return 1
        elif isinstance(kernel, MultiKernel):
            return kernel.max_length
        else:
            return kernel.length

    def _get_max_pre_samples(self) -> int:
        """Get maximum pre_samples for acausal kernel handling."""
        kernel = self.settings.kernel
        if kernel is None:
            return 0
        elif isinstance(kernel, MultiKernel):
            return kernel.max_pre_samples
        else:
            return kernel.pre_samples

    def _reset_state(self, message: AxisArray) -> None:
        """Initialize state for new input stream."""
        n_channels = message.data.shape[1] if message.data.ndim > 1 else 1
        max_overlap = self._get_max_kernel_length() - 1
        if max_overlap > 0:
            self._state.pending = np.zeros(
                (max_overlap, n_channels),
                dtype=self.settings.output_dtype,
            )
        else:
            self._state.pending = None
        self._state.pending_length = 0

    def _get_kernel_for_value(self, value: int | float) -> tuple[Kernel | None, float]:
        """
        Get kernel and scale factor for an event value.

        Returns:
            (kernel, scale) tuple.
        """
        kernel = self.settings.kernel
        scale = float(value) if self.settings.scale_by_value else 1.0

        if kernel is None:
            return None, scale
        elif isinstance(kernel, MultiKernel):
            return kernel.get(int(value)), scale
        else:
            return kernel, scale

    def _process(self, message: AxisArray) -> AxisArray:
        """Insert kernels at sparse event locations."""
        sparse_data = message.data
        n_samples = sparse_data.shape[0]
        n_channels = sparse_data.shape[1] if sparse_data.ndim > 1 else 1

        # Initialize output array
        output = np.zeros((n_samples, n_channels), dtype=self.settings.output_dtype)

        # Add pending contributions from previous chunk
        if self._state.pending is not None and self._state.pending_length > 0:
            overlap = min(self._state.pending_length, n_samples)
            output[:overlap] += self._state.pending[:overlap]

        # Reset pending for this chunk
        max_overlap = self._get_max_kernel_length() - 1
        if max_overlap > 0:
            if self._state.pending is None or self._state.pending.shape[0] < max_overlap:
                self._state.pending = np.zeros(
                    (max_overlap, n_channels),
                    dtype=self.settings.output_dtype,
                )
            else:
                self._state.pending[:] = 0
            self._state.pending_length = 0

        # Process each event
        if hasattr(sparse_data, "coords") and hasattr(sparse_data, "data"):
            # sparse.COO format
            coords = sparse_data.coords
            values = sparse_data.data

            for event_idx in range(len(values)):
                sample_idx = int(coords[0, event_idx])
                channel_idx = int(coords[1, event_idx]) if coords.shape[0] > 1 else 0
                value = values[event_idx]

                kernel, scale = self._get_kernel_for_value(value)

                if kernel is None:
                    # Unit impulse
                    if 0 <= sample_idx < n_samples:
                        output[sample_idx, channel_idx] += scale
                else:
                    # Insert kernel
                    pre = kernel.pre_samples
                    length = kernel.length

                    # Calculate kernel placement range
                    kernel_start = sample_idx - pre
                    kernel_end = kernel_start + length

                    # Portion within current chunk
                    chunk_start = max(0, kernel_start)
                    chunk_end = min(n_samples, kernel_end)

                    if chunk_start < chunk_end:
                        # Kernel indices for this portion
                        k_start = chunk_start - kernel_start
                        k_end = k_start + (chunk_end - chunk_start)

                        # Get kernel values
                        t = np.arange(k_start, k_end, dtype=np.float64) - pre
                        kernel_values = kernel.evaluate(t) * scale

                        output[chunk_start:chunk_end, channel_idx] += kernel_values

                    # Portion that extends into next chunk (pending)
                    if kernel_end > n_samples and self._state.pending is not None:
                        pending_start = max(0, n_samples - kernel_start)
                        pending_end = length
                        pending_samples = pending_end - pending_start

                        if pending_samples > 0:
                            # Kernel indices for pending portion
                            t = np.arange(pending_start, pending_end, dtype=np.float64) - pre
                            kernel_values = kernel.evaluate(t) * scale

                            # Add to pending buffer
                            self._state.pending[:pending_samples, channel_idx] += kernel_values
                            self._state.pending_length = max(
                                self._state.pending_length,
                                pending_samples,
                            )

        # Create output message
        return replace(message, data=output)


class SparseKernelInserterUnit(
    BaseTransformerUnit[
        SparseKernelInserterSettings,
        AxisArray,
        AxisArray,
        SparseKernelInserter,
    ]
):
    """Unit for SparseKernelInserter."""

    SETTINGS = SparseKernelInserterSettings
