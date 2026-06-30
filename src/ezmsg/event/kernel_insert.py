"""
Insert kernels at sparse event locations to produce dense signals.

This module provides efficient sparse-to-dense conversion by inserting
kernel waveforms at event locations. Overlapping kernels are summed.
"""

import ezmsg.core as ez
import numba
import numpy as np
import numpy.typing as npt
from ezmsg.baseproc import BaseStatefulTransformer, BaseTransformerUnit, processor_state
from ezmsg.util.messages.axisarray import AxisArray, replace

from .kernel import Kernel, MultiKernel


@numba.jit(nopython=True, cache=True)
def _insert_kernels_loop(
    output: np.ndarray,  # (n_samples, n_channels), accumulated in place
    pending: np.ndarray,  # (max_overlap, n_channels), accumulated in place
    event_samples: np.ndarray,  # (n_events,) int64 sample index of each event
    event_channels: np.ndarray,  # (n_events,) int64 channel index of each event
    event_rows: np.ndarray,  # (n_events,) int64 kernel-table row for each event
    event_scales: np.ndarray,  # (n_events,) float64 amplitude scale per event
    kernel_table: np.ndarray,  # (n_rows, max_len) float64 sampled kernels (zero-padded)
    kernel_lengths: np.ndarray,  # (n_rows,) int64 valid length of each kernel row
    kernel_pres: np.ndarray,  # (n_rows,) int64 pre_samples of each kernel row
    n_samples: int,
    max_overlap: int,
) -> int:
    """Scatter pre-sampled kernels onto a dense buffer at event locations.

    Compiled port of the per-event Python loop that used to live in
    :meth:`SparseKernelInserter._process`. Each kernel is materialized once into
    ``kernel_table`` (one zero-padded row per distinct kernel), so the only
    per-event work here is an indexed add. Output sample ``base + j`` (where
    ``base = sample - pre``) receives ``kernel_table[row, j] * scale``; samples
    that fall past the end of this chunk spill into ``pending`` for the next one.

    Returns the new ``pending_length`` (max filled index + 1, 0 if none).
    """
    pending_length = 0
    n_events = event_samples.shape[0]
    for e in range(n_events):
        ch = event_channels[e]
        row = event_rows[e]
        scale = event_scales[e]
        length = kernel_lengths[row]
        base = event_samples[e] - kernel_pres[row]

        for j in range(length):
            out_idx = base + j
            if out_idx < 0:
                continue
            if out_idx < n_samples:
                output[out_idx, ch] += kernel_table[row, j] * scale
            else:
                p = out_idx - n_samples
                if p < max_overlap:
                    pending[p, ch] += kernel_table[row, j] * scale
                    if p + 1 > pending_length:
                        pending_length = p + 1
    return pending_length


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

    # Pre-sampled kernels: one zero-padded row per distinct kernel, with each
    # row's valid length and pre_samples. Built once in _reset_state so the
    # hot loop only does indexed adds (see _insert_kernels_loop).
    kernel_table: npt.NDArray[np.float64] | None = None
    kernel_lengths: npt.NDArray[np.int64] | None = None
    kernel_pres: npt.NDArray[np.int64] | None = None

    # Map event value -> kernel_table row, with a fallback row for values not
    # present (mirrors MultiKernel.get's default-key behavior).
    value_to_row: dict | None = None
    default_row: int = 0


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

    def _build_kernel_table(self) -> None:
        """Pre-sample every kernel into a zero-padded table (built once).

        Each distinct kernel becomes one row, sampled at integer offsets so the
        compiled loop can place it with a plain indexed add: row ``j`` holds the
        kernel value that lands at output index ``(event_sample - pre) + j``.
        ``value_to_row`` maps an event value to its row, mirroring
        ``MultiKernel.get`` (unknown values fall back to ``default_row``).
        """
        kernel = self.settings.kernel

        def sample(k: Kernel) -> npt.NDArray[np.float64]:
            t = np.arange(k.length) - k.pre_samples
            return np.asarray(k.evaluate(t), dtype=np.float64)

        if kernel is None:
            # Unit impulse: a length-1 kernel of [1.0] at the event sample.
            kernels = [np.ones(1, dtype=np.float64)]
            pres = [0]
            value_to_row: dict = {}
            default_row = 0
        elif isinstance(kernel, MultiKernel):
            keys = list(kernel.keys)
            kernels = [sample(kernel[k]) for k in keys]
            pres = [kernel[k].pre_samples for k in keys]
            value_to_row = {int(k): i for i, k in enumerate(keys)}
            default_row = value_to_row[int(kernel._default_key)]
        else:
            kernels = [sample(kernel)]
            pres = [kernel.pre_samples]
            value_to_row = {}
            default_row = 0

        max_len = max(len(k) for k in kernels)
        table = np.zeros((len(kernels), max_len), dtype=np.float64)
        for i, k in enumerate(kernels):
            table[i, : len(k)] = k

        self._state.kernel_table = table
        self._state.kernel_lengths = np.array([len(k) for k in kernels], dtype=np.int64)
        self._state.kernel_pres = np.array(pres, dtype=np.int64)
        self._state.value_to_row = value_to_row
        self._state.default_row = default_row

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
        self._build_kernel_table()

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
        if hasattr(sparse_data, "coords") and hasattr(sparse_data, "data") and len(sparse_data.data) > 0:
            # sparse.COO format
            coords = sparse_data.coords
            values = sparse_data.data

            event_samples = np.ascontiguousarray(coords[0], dtype=np.int64)
            if coords.shape[0] > 1:
                event_channels = np.ascontiguousarray(coords[1], dtype=np.int64)
            else:
                event_channels = np.zeros(len(values), dtype=np.int64)

            # Resolve each event value to a kernel-table row (vectorized over the
            # typically-small set of distinct values), and its amplitude scale.
            int_values = values.astype(np.int64)
            uniq, inv = np.unique(int_values, return_inverse=True)
            value_to_row = self._state.value_to_row
            default_row = self._state.default_row
            uniq_rows = np.array(
                [value_to_row.get(int(v), default_row) for v in uniq],
                dtype=np.int64,
            )
            event_rows = uniq_rows[inv]

            if self.settings.scale_by_value:
                event_scales = values.astype(np.float64)
            else:
                event_scales = np.ones(len(values), dtype=np.float64)

            # numba needs a real array even when there is no pending buffer.
            pending = self._state.pending
            if pending is not None:
                pending_buf = pending
                max_overlap = pending.shape[0]
            else:
                pending_buf = np.zeros((0, n_channels), dtype=self.settings.output_dtype)
                max_overlap = 0

            pending_length = _insert_kernels_loop(
                output,
                pending_buf,
                event_samples,
                event_channels,
                event_rows,
                event_scales,
                self._state.kernel_table,
                self._state.kernel_lengths,
                self._state.kernel_pres,
                n_samples,
                max_overlap,
            )
            if pending is not None:
                self._state.pending_length = max(self._state.pending_length, pending_length)

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
