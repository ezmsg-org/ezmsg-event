"""
Kernel abstractions for sparse event processing.

Kernels can be applied to sparse events to produce either:
1. Dense signals (via SparseKernelInserter)
2. Binned activation features (via BinnedKernelActivation)
"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import numpy.typing as npt


class Kernel(ABC):
    """
    Base class for kernels applied to sparse events.

    A kernel defines a shape that gets inserted/convolved at event locations.
    Supports both causal (forward-looking) and acausal (symmetric) kernels.
    """

    @property
    @abstractmethod
    def length(self) -> int:
        """Total kernel length in samples."""
        pass

    @property
    @abstractmethod
    def pre_samples(self) -> int:
        """Number of samples before t=0 (for acausal kernels)."""
        pass

    @abstractmethod
    def evaluate(self, t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        Evaluate kernel at time offsets t (in samples relative to event).

        Args:
            t: Array of time offsets in samples. t=0 is the event time.

        Returns:
            Kernel values at the given offsets.
        """
        pass

    @property
    def is_causal(self) -> bool:
        """True if kernel is zero for t < 0."""
        return self.pre_samples == 0

    @property
    def post_samples(self) -> int:
        """Number of samples at and after t=0."""
        return self.length - self.pre_samples


class ArrayKernel(Kernel):
    """
    Kernel from explicit array (e.g., spike waveforms).

    Args:
        data: 1D array of kernel values.
        pre_samples: Number of samples before t=0. Default 0 (causal kernel).
            For a waveform centered at t=0, use pre_samples = len(data) // 2.
    """

    def __init__(self, data: npt.NDArray, pre_samples: int = 0):
        self._data = np.asarray(data, dtype=np.float64)
        if self._data.ndim != 1:
            raise ValueError("Kernel data must be 1-dimensional")
        self._pre_samples = pre_samples
        if pre_samples < 0 or pre_samples > len(self._data):
            raise ValueError(f"pre_samples must be in [0, {len(self._data)}]")

    @property
    def length(self) -> int:
        return len(self._data)

    @property
    def pre_samples(self) -> int:
        return self._pre_samples

    @property
    def data(self) -> npt.NDArray[np.float64]:
        """Raw kernel data array."""
        return self._data

    def evaluate(self, t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Evaluate kernel at time offsets."""
        t = np.asarray(t)
        indices = (t + self._pre_samples).astype(int)
        valid = (indices >= 0) & (indices < len(self._data))
        result = np.zeros(t.shape, dtype=self._data.dtype)
        result[valid] = self._data[indices[valid]]
        return result


class FunctionalKernel(Kernel):
    """
    Kernel from a function (e.g., exponential decay, Gaussian).

    The function should accept (t, sigma) where t is time in samples
    and sigma is the time constant in samples.

    Args:
        func: Kernel function f(t, sigma) -> values.
        sigma: Time constant in seconds.
        fs: Sample rate in Hz (for converting sigma to samples).
        truncate_at: Truncate kernel at this many time constants. Default 5.0.
        causal: If True, kernel is zero for t < 0. Default True.

    Example:
        # Exponential decay kernel with 10ms time constant
        kernel = FunctionalKernel(
            func=lambda t, s: (t >= 0) * np.exp(-t / s) / s,
            sigma=0.010,  # 10ms
            fs=30000,
        )
    """

    def __init__(
        self,
        func: Callable[[npt.NDArray, float], npt.NDArray],
        sigma: float,
        fs: float,
        truncate_at: float = 5.0,
        causal: bool = True,
    ):
        self._func = func
        self._sigma = sigma
        self._fs = fs
        self._sigma_samples = sigma * fs
        self._causal = causal

        if causal:
            self._pre = 0
            self._length = max(1, int(truncate_at * self._sigma_samples))
        else:
            # Symmetric kernel
            half_length = max(1, int(truncate_at * self._sigma_samples))
            self._pre = half_length
            self._length = 2 * half_length + 1

    @property
    def length(self) -> int:
        return self._length

    @property
    def pre_samples(self) -> int:
        return self._pre

    @property
    def sigma(self) -> float:
        """Time constant in seconds."""
        return self._sigma

    @property
    def sigma_samples(self) -> float:
        """Time constant in samples."""
        return self._sigma_samples

    def evaluate(self, t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Evaluate kernel function at time offsets."""
        return self._func(np.asarray(t, dtype=np.float64), self._sigma_samples)


class MultiKernel:
    """
    Dictionary of kernels indexed by event value.

    Useful when different event types (e.g., waveform IDs 1, 2, 3)
    should produce different kernel shapes.

    Args:
        kernels: Dictionary mapping event values to Kernel objects.
        default_key: Key to use for unknown event values. Default is first key.
    """

    def __init__(self, kernels: dict[int, Kernel], default_key: int | None = None):
        if not kernels:
            raise ValueError("kernels dict cannot be empty")
        self._kernels = kernels
        self._default_key = default_key if default_key is not None else next(iter(kernels))

    def get(self, value: int) -> Kernel:
        """Get kernel for event value, falling back to default."""
        return self._kernels.get(value, self._kernels[self._default_key])

    def __getitem__(self, value: int) -> Kernel:
        """Get kernel for event value (raises KeyError if not found)."""
        return self._kernels[value]

    def __contains__(self, value: int) -> bool:
        return value in self._kernels

    @property
    def max_length(self) -> int:
        """Maximum kernel length across all kernels."""
        return max(k.length for k in self._kernels.values())

    @property
    def max_pre_samples(self) -> int:
        """Maximum pre_samples across all kernels."""
        return max(k.pre_samples for k in self._kernels.values())

    @property
    def max_post_samples(self) -> int:
        """Maximum post_samples across all kernels."""
        return max(k.post_samples for k in self._kernels.values())

    @property
    def keys(self) -> list[int]:
        """Available kernel keys."""
        return list(self._kernels.keys())


# =============================================================================
# Common kernel functions
# =============================================================================


def exponential_kernel(t: npt.NDArray, sigma: float) -> npt.NDArray:
    """
    Causal exponential decay kernel: k(t) = exp(-t/sigma) / sigma for t >= 0.

    Normalized so that integral from 0 to inf equals 1.
    """
    t = np.asarray(t)
    result = np.zeros_like(t, dtype=np.float64)
    valid = t >= 0
    result[valid] = np.exp(-t[valid] / sigma) / sigma
    return result


def alpha_kernel(t: npt.NDArray, sigma: float) -> npt.NDArray:
    """
    Alpha function kernel: k(t) = (t/sigma^2) * exp(-t/sigma) for t >= 0.

    Peaks at t = sigma. Normalized so that integral from 0 to inf equals 1.
    """
    t = np.asarray(t)
    result = np.zeros_like(t, dtype=np.float64)
    valid = t >= 0
    result[valid] = (t[valid] / sigma**2) * np.exp(-t[valid] / sigma)
    return result


def gaussian_kernel(t: npt.NDArray, sigma: float) -> npt.NDArray:
    """
    Gaussian kernel: k(t) = exp(-t^2 / (2*sigma^2)) / (sigma * sqrt(2*pi)).

    Symmetric (acausal). Normalized so that integral equals 1.
    """
    t = np.asarray(t)
    return np.exp(-(t**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def boxcar_kernel(t: npt.NDArray, sigma: float) -> npt.NDArray:
    """
    Boxcar (rectangular) kernel: k(t) = 1/(2*sigma) for |t| < sigma.

    Symmetric (acausal). Width is 2*sigma. Normalized so that integral equals 1.
    """
    t = np.asarray(t)
    half_width = sigma
    result = np.zeros_like(t, dtype=np.float64)
    valid = np.abs(t) < half_width
    result[valid] = 0.5 / half_width
    return result


def causal_boxcar_kernel(t: npt.NDArray, sigma: float) -> npt.NDArray:
    """
    Causal boxcar kernel: k(t) = 1/sigma for 0 <= t < sigma.

    Normalized so that integral equals 1.
    """
    t = np.asarray(t)
    result = np.zeros_like(t, dtype=np.float64)
    valid = (t >= 0) & (t < sigma)
    result[valid] = 1.0 / sigma
    return result
