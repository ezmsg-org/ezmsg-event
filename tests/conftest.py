"""Shared test fixtures and helpers for ezmsg-event tests."""

import numpy as np
import sparse
from ezmsg.util.messages.axisarray import AxisArray

FS = 30_000.0
N_CH = 4
CHUNK_LEN = 600  # 20ms at 30kHz


def make_dense_msg(n_time: int, n_ch: int = N_CH, fs: float = FS, offset: float = 0.0) -> AxisArray:
    """Create a dense AxisArray message with random data."""
    return AxisArray(
        data=np.random.randn(n_time, n_ch),
        dims=["time", "ch"],
        axes={"time": AxisArray.Axis.TimeAxis(fs=fs, offset=offset)},
    )


def make_sparse_event_msg(
    n_time: int, n_ch: int = N_CH, fs: float = FS, density: float = 0.01, offset: float = 0.0
) -> AxisArray:
    """Create a sparse AxisArray message (COO format)."""
    if n_time == 0:
        s = sparse.COO(coords=np.zeros((2, 0), dtype=np.int64), data=np.zeros(0, dtype=bool), shape=(0, n_ch))
    else:
        s = sparse.random((n_time, n_ch), density=density, random_state=np.random.default_rng(42)) > 0
    return AxisArray(
        data=s,
        dims=["time", "ch"],
        axes={"time": AxisArray.Axis.TimeAxis(fs=fs, offset=offset)},
    )


def make_rate_msg(
    n_time: int, n_ch: int = N_CH, rate_hz: float = 50.0, bin_fs: float = 50.0, offset: float = 0.0
) -> AxisArray:
    """Create a dense AxisArray with firing-rate data (for PoissonEventTransformer)."""
    return AxisArray(
        data=np.full((n_time, n_ch), rate_hz),
        dims=["time", "ch"],
        axes={"time": AxisArray.Axis.TimeAxis(fs=bin_fs, offset=offset)},
    )
