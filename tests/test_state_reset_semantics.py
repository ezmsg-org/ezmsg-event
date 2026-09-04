"""When these transformers rebuild their state, now that the base class decides.

``ThresholdCrossingTransformer`` and ``RefractoryTransformer`` no longer
implement ``_hash_message``: the base class folds in the message key, the dims,
the length of every dimension except the one the stream is chunked along, the
coordinate values on those dimensions, and the gain and offset of any linear
axis among them.

The change that matters is the channel *fingerprint*. A source that renames or
reorders its channels without changing how many it sends -- a device
reconfigured mid-session, a montage swapped -- previously looked identical to
these transformers, so per-channel state carried over onto channels it did not
belong to. Nothing about the output announces that; these tests are what would
catch it coming back.

``BinnedKernelActivation`` still overrides, because dtype and array backend are
things the default cannot see. Its tests here pin both halves: what it adds, and
what it now inherits.
"""

import numpy as np
import pytest
import sparse
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

from ezmsg.event.kernel_activation import (
    ActivationKernelType,
    BinnedKernelActivation,
    BinnedKernelActivationSettings,
)
from ezmsg.event.peak import ThresholdCrossingTransformer, ThresholdSettings
from ezmsg.event.refractory import RefractorySettings, RefractoryTransformer

FS = 30_000.0


def dense_msg(labels, n_time=600, fs=FS, key="dev", seed=0):
    """A dense message carrying a *labelled* channel axis.

    The conftest helpers omit the ``ch`` axis, which is fine for what they test
    but makes a relabel invisible by construction -- there is nothing to relabel.
    A real source (blackrock, lsl, a replayed file) always attaches one.
    """
    rng = np.random.default_rng(seed)
    return AxisArray(
        data=rng.standard_normal((n_time, len(labels))),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs),
            "ch": CoordinateAxis(data=np.array(labels), dims=["ch"]),
        },
        key=key,
        chunk_dim="time",
    )


def sparse_msg(labels, n_time=600, fs=FS, key="dev", dtype=bool):
    coords = np.array(
        [[i * 7 % n_time for i in range(len(labels) * 3)], [i % len(labels) for i in range(len(labels) * 3)]]
    )
    data = np.ones(coords.shape[1], dtype=dtype)
    return AxisArray(
        data=sparse.COO(coords, data, shape=(n_time, len(labels))),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs),
            "ch": CoordinateAxis(data=np.array(labels), dims=["ch"]),
        },
        key=key,
        chunk_dim="time",
    )


def at_sample(labels, sample_ix, n_time=600, fs=FS, key="dev"):
    """One event on every channel, at a chosen sample index."""
    n_ch = len(labels)
    coords = np.array([[sample_ix] * n_ch, list(range(n_ch))])
    return AxisArray(
        data=sparse.COO(coords, np.ones(n_ch, dtype=bool), shape=(n_time, n_ch)),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs),
            "ch": CoordinateAxis(data=np.array(labels), dims=["ch"]),
        },
        key=key,
        chunk_dim="time",
    )


ARM_A = ["armA-1", "armA-2", "armA-3", "armA-4"]
ARM_B = ["armB-1", "armB-2", "armB-3", "armB-4"]


class TestARelabelIsNoticed:
    """Same channel count, different channels. The case that used to be silent."""

    def test_threshold_crossing_rebuilds(self):
        proc = ThresholdCrossingTransformer(ThresholdSettings(threshold=-1.0, auto_scale_tau=0.5))
        proc(dense_msg(ARM_A))
        before = proc._hash
        proc(dense_msg(ARM_B, seed=1))
        assert proc._hash != before

    def test_the_adaptive_scaler_does_not_carry_over(self):
        """`auto_scale_tau` gives the transformer a per-channel z-scoring state.
        Carrying armA's running mean onto armB is the same failure as a filter
        keeping the previous device's history."""
        settings = ThresholdSettings(threshold=-1.0, auto_scale_tau=0.5)

        carried = ThresholdCrossingTransformer(settings)
        carried(dense_msg(ARM_A, seed=0))
        got = carried(dense_msg(ARM_B, seed=1))

        clean = ThresholdCrossingTransformer(settings)
        want = clean(dense_msg(ARM_B, seed=1))

        assert (got.data != want.data).nnz == 0, "state survived a channel relabel"

    def test_refractory_rebuilds(self):
        proc = RefractoryTransformer(RefractorySettings(dur=0.002))
        proc(sparse_msg(ARM_A))
        before = proc._hash
        proc(sparse_msg(ARM_B))
        assert proc._hash != before

    def test_refractory_elapsed_counters_do_not_carry_over(self):
        """The refractory window has to be straddled for this to mean anything.

        armA fires on its *last* sample and armB on its *sixth*; at 30 kHz a
        2 ms window is 60 samples, so armA's `elapsed` counters -- if they
        survived -- would still be inside the window and would swallow armB's
        events entirely. Give both streams the same event positions and the
        carried state changes nothing, and the test passes either way.
        """
        settings = RefractorySettings(dur=0.002)
        n_time = 600
        late = at_sample(ARM_A, n_time - 1, n_time=n_time)
        early = at_sample(ARM_B, 5, n_time=n_time)

        carried = RefractoryTransformer(settings)
        carried(late)
        got = carried(early)

        clean = RefractoryTransformer(settings)
        want = clean(at_sample(ARM_B, 5, n_time=n_time))

        assert want.data.nnz == len(ARM_B), "the reference run should pass every event"
        np.testing.assert_array_equal(got.data.todense(), want.data.todense())


class TestWhatMustNotResetStillDoesNot:
    def test_chunk_size_jitter_is_ignored(self):
        """The chunk dimension's length is whatever arrived; rebuilding on it
        would throw away the trailing buffer on every irregular chunk."""
        proc = ThresholdCrossingTransformer(ThresholdSettings(threshold=-1.0))
        proc(dense_msg(ARM_A, n_time=600))
        before = proc._hash
        proc(dense_msg(ARM_A, n_time=317))
        assert proc._hash == before

    def test_the_same_channels_again_is_not_a_change(self):
        proc = RefractoryTransformer(RefractorySettings(dur=0.002))
        proc(sparse_msg(ARM_A))
        before = proc._hash
        proc(sparse_msg(ARM_A))
        assert proc._hash == before


class TestKernelActivationKeepsItsOverride:
    """It adds dtype and backend, which the base class cannot see, and now
    inherits the key and fingerprint, which it previously ignored."""

    @staticmethod
    def _proc():
        return BinnedKernelActivation(
            BinnedKernelActivationSettings(kernel_type=ActivationKernelType.EXPONENTIAL, tau=0.05, bin_duration=0.02)
        )

    def test_a_dtype_change_still_resets(self):
        proc = self._proc()
        proc(sparse_msg(ARM_A, dtype=bool))
        before = proc._hash
        proc(sparse_msg(ARM_A, dtype=np.float32))
        assert proc._hash != before

    def test_a_relabel_now_resets(self):
        """New: the previous hash saw only ndim, dtype, channel count and rate."""
        proc = self._proc()
        proc(sparse_msg(ARM_A))
        before = proc._hash
        proc(sparse_msg(ARM_B))
        assert proc._hash != before

    def test_a_stream_switch_now_resets(self):
        """Also new. `activation`, `dense_carry` and `samples_since_update` are
        per-channel running state and must not follow a key change."""
        proc = self._proc()
        proc(sparse_msg(ARM_A, key="deviceA"))
        before = proc._hash
        proc(sparse_msg(ARM_A, key="deviceB"))
        assert proc._hash != before

    def test_chunk_size_jitter_is_still_ignored(self):
        proc = self._proc()
        proc(sparse_msg(ARM_A, n_time=600))
        before = proc._hash
        proc(sparse_msg(ARM_A, n_time=317))
        assert proc._hash == before

    def test_a_missing_time_axis_is_still_rejected(self):
        proc = self._proc()
        msg = AxisArray(
            data=sparse.COO(np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=bool), shape=(10, 4)),
            dims=["time", "ch"],
            axes={"ch": CoordinateAxis(data=np.array(ARM_A), dims=["ch"])},
            key="dev",
        )
        with pytest.raises(ValueError, match="sample rate"):
            proc(msg)
