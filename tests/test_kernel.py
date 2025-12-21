"""Unit tests for ezmsg.event.kernel module."""

import numpy as np
import pytest

from ezmsg.event.kernel import (
    ArrayKernel,
    FunctionalKernel,
    MultiKernel,
    alpha_kernel,
    boxcar_kernel,
    causal_boxcar_kernel,
    exponential_kernel,
    gaussian_kernel,
)


class TestArrayKernel:
    """Tests for ArrayKernel."""

    def test_basic_creation(self):
        """Create array kernel with default settings."""
        data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        kernel = ArrayKernel(data)

        assert kernel.length == 5
        assert kernel.pre_samples == 0
        assert kernel.post_samples == 5
        assert kernel.is_causal is True

    def test_acausal_kernel(self):
        """Create acausal (symmetric) kernel."""
        data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        kernel = ArrayKernel(data, pre_samples=2)

        assert kernel.length == 5
        assert kernel.pre_samples == 2
        assert kernel.post_samples == 3
        assert kernel.is_causal is False

    def test_evaluate_causal(self):
        """Evaluate causal kernel at various offsets."""
        data = np.array([10.0, 20.0, 30.0])
        kernel = ArrayKernel(data)

        # Evaluate at kernel sample points
        t = np.array([0, 1, 2])
        values = kernel.evaluate(t.astype(float))
        np.testing.assert_array_equal(values, [10.0, 20.0, 30.0])

        # Evaluate outside range
        t = np.array([-1, 3])
        values = kernel.evaluate(t.astype(float))
        np.testing.assert_array_equal(values, [0.0, 0.0])

    def test_evaluate_acausal(self):
        """Evaluate acausal kernel at various offsets."""
        data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        kernel = ArrayKernel(data, pre_samples=2)

        # t=0 should map to center (index 2)
        t = np.array([-2, -1, 0, 1, 2])
        values = kernel.evaluate(t.astype(float))
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0, 2.0, 1.0])

    def test_invalid_pre_samples(self):
        """Reject invalid pre_samples values."""
        data = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError):
            ArrayKernel(data, pre_samples=-1)

        with pytest.raises(ValueError):
            ArrayKernel(data, pre_samples=4)

    def test_non_1d_data(self):
        """Reject non-1D data."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError):
            ArrayKernel(data)


class TestFunctionalKernel:
    """Tests for FunctionalKernel."""

    def test_exponential_kernel(self):
        """Create exponential decay kernel."""
        kernel = FunctionalKernel(
            func=exponential_kernel,
            sigma=0.010,  # 10ms
            fs=1000,  # 1kHz
            truncate_at=5.0,
            causal=True,
        )

        assert kernel.length == 50  # 5 * 10 samples
        assert kernel.pre_samples == 0
        assert kernel.is_causal is True
        assert kernel.sigma == 0.010
        assert kernel.sigma_samples == 10.0

    def test_gaussian_kernel(self):
        """Create Gaussian (acausal) kernel."""
        kernel = FunctionalKernel(
            func=gaussian_kernel,
            sigma=0.020,  # 20ms
            fs=1000,
            truncate_at=3.0,
            causal=False,
        )

        assert kernel.pre_samples == 60  # 3 * 20 samples
        assert kernel.is_causal is False
        # length should be 2 * 60 + 1 = 121
        assert kernel.length == 121

    def test_evaluate_exponential(self):
        """Evaluate exponential kernel."""
        kernel = FunctionalKernel(
            func=exponential_kernel,
            sigma=0.010,
            fs=1000,
        )

        # At t=0, exponential should be 1/sigma_samples
        t = np.array([0.0])
        values = kernel.evaluate(t)
        expected = 1.0 / 10.0  # 1/sigma_samples
        assert values[0] == pytest.approx(expected)

        # Negative t should be 0
        t = np.array([-5.0])
        values = kernel.evaluate(t)
        assert values[0] == 0.0


class TestMultiKernel:
    """Tests for MultiKernel."""

    def test_basic_creation(self):
        """Create multi-kernel with multiple waveforms."""
        k1 = ArrayKernel(np.array([1.0, 2.0, 1.0]))
        k2 = ArrayKernel(np.array([2.0, 4.0, 2.0]))
        k3 = ArrayKernel(np.array([1.0, 1.0, 1.0, 1.0]))

        multi = MultiKernel({1: k1, 2: k2, 3: k3})

        assert multi.max_length == 4
        assert 1 in multi
        assert 2 in multi
        assert 3 in multi
        assert 4 not in multi

    def test_get_kernel(self):
        """Get kernels by value."""
        k1 = ArrayKernel(np.array([1.0]))
        k2 = ArrayKernel(np.array([2.0]))

        multi = MultiKernel({1: k1, 2: k2})

        assert multi.get(1) is k1
        assert multi.get(2) is k2
        assert multi.get(99) is k1  # Default to first

    def test_getitem(self):
        """Access kernels via subscript."""
        k1 = ArrayKernel(np.array([1.0]))
        k2 = ArrayKernel(np.array([2.0]))

        multi = MultiKernel({1: k1, 2: k2})

        assert multi[1] is k1
        assert multi[2] is k2

        with pytest.raises(KeyError):
            _ = multi[99]

    def test_empty_dict_rejected(self):
        """Reject empty kernel dict."""
        with pytest.raises(ValueError):
            MultiKernel({})


class TestKernelFunctions:
    """Tests for kernel function implementations."""

    def test_exponential_integral(self):
        """Exponential kernel integrates to approximately 1."""
        sigma = 10.0  # samples
        t = np.arange(0, 200)  # 20 time constants for better convergence
        values = exponential_kernel(t, sigma)

        # Numerical integration (should be close to 1)
        # Note: discrete sum approximates integral, slight overestimate expected
        integral = np.sum(values)
        assert integral == pytest.approx(1.0, rel=0.1)

    def test_alpha_peak(self):
        """Alpha kernel peaks at sigma."""
        sigma = 20.0  # samples
        t = np.arange(0, 200)
        values = alpha_kernel(t, sigma)

        peak_idx = np.argmax(values)
        assert peak_idx == pytest.approx(sigma, abs=1)

    def test_gaussian_symmetric(self):
        """Gaussian kernel is symmetric."""
        sigma = 10.0
        t = np.arange(-50, 51)
        values = gaussian_kernel(t, sigma)

        # Check symmetry
        np.testing.assert_array_almost_equal(values, values[::-1])

    def test_boxcar_width(self):
        """Boxcar kernel has correct width."""
        sigma = 10.0
        t = np.arange(-20, 21)
        values = boxcar_kernel(t, sigma)

        # Non-zero values should be within [-sigma, sigma)
        nonzero_t = t[values > 0]
        assert np.all(np.abs(nonzero_t) < sigma)

    def test_causal_boxcar(self):
        """Causal boxcar is zero for t < 0."""
        sigma = 10.0
        t = np.arange(-5, 15)
        values = causal_boxcar_kernel(t, sigma)

        # Should be zero for t < 0
        assert np.all(values[t < 0] == 0)
        # Should be non-zero for 0 <= t < sigma
        assert np.all(values[(t >= 0) & (t < sigma)] > 0)
        # Should be zero for t >= sigma
        assert np.all(values[t >= sigma] == 0)
