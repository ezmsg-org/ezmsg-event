"""Unit tests for ezmsg.event.kernel_activation module."""

import numpy as np
import pytest
import sparse
from conftest import CHUNK_LEN, FS, N_CH, make_sparse_event_msg
from ezmsg.util.messages.axisarray import AxisArray, replace

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


class TestEmptyTime:
    """Test handling of length-0 time dimension inputs."""

    @pytest.mark.parametrize(
        "kernel_type",
        [ActivationKernelType.EXPONENTIAL, ActivationKernelType.ALPHA, ActivationKernelType.COUNT],
    )
    def test_empty_time_after_init(self, kernel_type: ActivationKernelType):
        """Normal → empty → normal: mid-stream empty message."""
        proc = BinnedKernelActivation(BinnedKernelActivationSettings(kernel_type=kernel_type, bin_duration=0.02))
        msg1 = make_sparse_event_msg(CHUNK_LEN, offset=0.0)
        msg_empty = make_sparse_event_msg(0, offset=CHUNK_LEN / FS)
        msg2 = make_sparse_event_msg(CHUNK_LEN, offset=CHUNK_LEN / FS)

        out1 = proc(msg1)
        assert out1.data.ndim == 2

        out_empty = proc(msg_empty)
        assert out_empty.data.shape[0] == 0 or out_empty.data.ndim == 2

        out2 = proc(msg2)
        assert out2.data.ndim == 2
        assert out2.data.shape[1] == N_CH

    @pytest.mark.parametrize(
        "kernel_type",
        [ActivationKernelType.EXPONENTIAL, ActivationKernelType.ALPHA, ActivationKernelType.COUNT],
    )
    def test_empty_time_first(self, kernel_type: ActivationKernelType):
        """Empty → normal: empty first message triggers _reset_state on empty data."""
        proc = BinnedKernelActivation(BinnedKernelActivationSettings(kernel_type=kernel_type, bin_duration=0.02))
        msg_empty = make_sparse_event_msg(0, offset=0.0)
        msg_normal = make_sparse_event_msg(CHUNK_LEN, offset=0.0)

        out_empty = proc(msg_empty)
        assert out_empty.data.ndim == 2

        out_normal = proc(msg_normal)
        assert out_normal.data.ndim == 2
        assert out_normal.data.shape[1] == N_CH


def _kernel_reference(events: np.ndarray, kernel_type, tau: float, fs: float, normalize: bool) -> np.ndarray:
    """Activation at every sample by direct convolution of a dense ``(time, ch)`` event array."""
    n = np.arange(events.shape[0])
    tau_s = tau * fs
    kernel = np.exp(-n / tau_s) if kernel_type == ActivationKernelType.EXPONENTIAL else (n / tau_s) * np.exp(-n / tau_s)
    if normalize:
        kernel = kernel / tau_s
    return np.stack([np.convolve(events[:, c], kernel)[: len(n)] for c in range(events.shape[1])], axis=1)


class TestMaxAggregation:
    """MAX is the largest activation at any sample of the bin, chunked or not."""

    FS = 1000.0
    SPB = 10  # bin_duration * FS

    def _events(self, scale_by_value: bool) -> np.ndarray:
        rng = np.random.default_rng(3)
        events = (rng.random((200, 2)) < 0.04).astype(float)
        if scale_by_value:
            events *= rng.normal(size=events.shape)  # mixed signs: the max may sit at a bin's end
        return events

    def _proc(self, kernel_type, scale_by_value=False, normalize=False):
        return BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=kernel_type,
                tau=0.008,
                bin_duration=self.SPB / self.FS,
                aggregation=BinAggregation.MAX,
                scale_by_value=scale_by_value,
                normalize=normalize,
            )
        )

    def _msg(self, data, offset_samples: int, dense: bool) -> AxisArray:
        return AxisArray(
            data=data if dense else sparse.COO.from_numpy(data),
            dims=["time", "ch"],
            axes={"time": AxisArray.TimeAxis(fs=self.FS, offset=offset_samples / self.FS)},
        )

    def _run(self, proc, events, chunk: int, dense: bool) -> np.ndarray:
        outs = [proc(self._msg(events[i : i + chunk], i, dense)).data for i in range(0, events.shape[0], chunk)]
        return np.concatenate(outs, axis=0)

    @pytest.mark.parametrize("kernel_type", [ActivationKernelType.EXPONENTIAL, ActivationKernelType.ALPHA])
    @pytest.mark.parametrize("dense", [False, True], ids=["sparse", "dense"])
    @pytest.mark.parametrize("chunk", [200, 7, 23], ids=["one_chunk", "sub_bin_chunks", "chunks_across_bins"])
    @pytest.mark.parametrize("scale_by_value", [False, True], ids=["count", "weighted"])
    def test_matches_dense_reference(self, kernel_type, dense, chunk, scale_by_value):
        events = self._events(scale_by_value)
        ref = _kernel_reference(events, kernel_type, 0.008, self.FS, normalize=True)
        expected = ref.reshape(-1, self.SPB, ref.shape[1]).max(axis=1)
        out = self._run(self._proc(kernel_type, scale_by_value, normalize=True), events, chunk, dense)
        np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-12)

    def test_mlx_dense_input_matches_numpy(self):
        mx = pytest.importorskip("mlx.core")
        events = self._events(False)
        out_np = self._proc(ActivationKernelType.ALPHA)(self._msg(events, 0, dense=True)).data
        out_mx = self._proc(ActivationKernelType.ALPHA)(self._msg(mx.array(events), 0, dense=True)).data
        np.testing.assert_allclose(np.asarray(out_mx), out_np, rtol=1e-6)

    def test_count_kernel_refuses_max(self):
        with pytest.raises(ValueError, match="MAX"):
            self._proc(ActivationKernelType.COUNT)(self._msg(self._events(False), 0, dense=False))

    def _labelled(self, data, offset_samples: int, dense: bool, segment: int) -> AxisArray:
        return replace(self._msg(data, offset_samples, dense), segment=segment, attrs={"unit": "uV"})

    @pytest.mark.parametrize("kernel_type", [ActivationKernelType.EXPONENTIAL, ActivationKernelType.ALPHA])
    @pytest.mark.parametrize("dense", [False, True], ids=["sparse", "dense"])
    @pytest.mark.parametrize("chunk", [206, 7], ids=["one_chunk", "chunked"])
    def test_flush_emits_the_partial_bin_max(self, kernel_type, dense, chunk):
        """At a gap, the open bin's max over the samples it got is emitted, at the bin's place on the grid."""
        events = self._events(False)
        events = np.concatenate((events, events[:6]))  # 20 full bins and 6 samples of a 21st
        ref = _kernel_reference(events, kernel_type, 0.008, self.FS, normalize=True)
        proc = self._proc(kernel_type, normalize=True)
        outs = [proc(self._labelled(events[i : i + chunk], i, dense, 3)) for i in range(0, events.shape[0], chunk)]
        tail = proc.flush()

        np.testing.assert_allclose(tail.data, ref[200:].max(axis=0, keepdims=True), rtol=1e-9, atol=1e-12)
        assert tail.data.dtype == outs[-1].data.dtype and tail.dims == outs[-1].dims
        assert tail.axes["time"].offset == pytest.approx(200 / self.FS)
        assert tail.axes["time"].gain == pytest.approx(self.SPB / self.FS)
        assert (tail.segment, tail.config_id, tail.attrs) == (3, outs[-1].config_id, {"unit": "uV"})
        assert proc.flush() is None

    def test_flush_emits_nothing_for_an_empty_partial_bin(self):
        proc = self._proc(ActivationKernelType.ALPHA)
        proc(self._msg(self._events(False), 0, dense=False))
        assert proc.flush() is None

    @pytest.mark.parametrize("kernel_type", [ActivationKernelType.EXPONENTIAL, ActivationKernelType.ALPHA])
    def test_after_a_flush_the_next_segment_starts_clean(self, kernel_type):
        events = self._events(False)
        proc = self._proc(kernel_type)
        proc(self._labelled(events[:206], 0, False, 0))
        proc.flush()
        nxt = self._labelled(events, 5000, False, 1)
        continued = proc(nxt)
        fresh = self._proc(kernel_type)(nxt)
        np.testing.assert_array_equal(continued.data, fresh.data)
        assert continued.axes["time"] == fresh.axes["time"]


class TestAlphaIsExact:
    """The alpha activation is (t/tau) exp(-t/tau) whatever the bin width it is sampled at."""

    @pytest.mark.parametrize("bin_duration", [0.005, 0.010, 0.050])
    def test_single_event_follows_the_kernel(self, bin_duration):
        proc = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.ALPHA,
                tau=0.020,
                bin_duration=bin_duration,
                aggregation=BinAggregation.LAST,
                normalize=False,
            )
        )
        out = proc(make_sparse_message(coords=[(0, 0)], values=[1], shape=(200, 1)))
        t = (np.arange(out.data.shape[0]) + 1) * bin_duration / 0.020
        np.testing.assert_allclose(out.data[:, 0], t * np.exp(-t), rtol=1e-9)


@pytest.mark.parametrize("kernel_type", [ActivationKernelType.EXPONENTIAL, ActivationKernelType.ALPHA])
def test_last_is_chunk_invariant_with_sub_bin_chunks(kernel_type):
    """A chunk too short to close a bin must still advance the decay clock."""
    events = (np.random.default_rng(5).random((200, 2)) < 0.04).astype(float)

    def run(chunk):
        proc = BinnedKernelActivation(
            BinnedKernelActivationSettings(kernel_type=kernel_type, tau=0.008, bin_duration=0.010)
        )
        return np.concatenate(
            [
                proc(
                    AxisArray(
                        sparse.COO.from_numpy(events[i : i + chunk]),
                        dims=["time", "ch"],
                        axes={"time": AxisArray.TimeAxis(fs=1000.0, offset=i / 1000.0)},
                    )
                ).data
                for i in range(0, 200, chunk)
            ]
        )

    np.testing.assert_allclose(run(7), run(200), rtol=1e-9)


class TestKernelSum:
    """SUM with a kernel is the per-sample activation summed over each bin's samples."""

    FS = 1000.0

    AGGREGATION = BinAggregation.SUM

    def _run(self, kernel_type, events, chunks, bin_duration=0.010, scale_by_value=False) -> np.ndarray:
        proc = BinnedKernelActivation(
            BinnedKernelActivationSettings(
                kernel_type=kernel_type,
                tau=0.008,
                bin_duration=bin_duration,
                aggregation=self.AGGREGATION,
                scale_by_value=scale_by_value,
            )
        )
        starts = np.cumsum([0, *chunks])
        assert starts[-1] == events.shape[0]
        outs = [
            proc(
                AxisArray(
                    sparse.COO.from_numpy(events[a:b]),
                    dims=["time", "ch"],
                    axes={"time": AxisArray.TimeAxis(fs=self.FS, offset=a / self.FS)},
                )
            ).data
            for a, b in zip(starts[:-1], starts[1:])
        ]
        return np.concatenate(outs, axis=0)

    @pytest.mark.parametrize("kernel_type", [ActivationKernelType.EXPONENTIAL, ActivationKernelType.ALPHA])
    @pytest.mark.parametrize("scale_by_value", [False, True], ids=["count", "weighted"])
    def test_matches_dense_reference(self, kernel_type, scale_by_value):
        rng = np.random.default_rng(7)
        events = (rng.random((200, 2)) < 0.05).astype(float)
        if scale_by_value:
            events *= rng.normal(size=events.shape)
        ref = _kernel_reference(events, kernel_type, 0.008, self.FS, normalize=True)
        expected = ref.reshape(-1, 10, 2).sum(axis=1)
        out = self._run(kernel_type, events, [200], scale_by_value=scale_by_value)
        np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("kernel_type", [ActivationKernelType.EXPONENTIAL, ActivationKernelType.ALPHA])
    @pytest.mark.parametrize("bin_duration", [0.010, 0.0075], ids=["whole", "fractional"])
    def test_irregular_chunks_match_one_chunk(self, kernel_type, bin_duration):
        """Chunks shorter than a bin, and bins split over several chunks, lose no events."""
        events = (np.random.default_rng(8).random((200, 2)) < 0.05).astype(float)
        chunks = [3, 1, 4, 1, 5, 9, 2, 6, 25, 3, 50, 7, 1, 83]
        np.testing.assert_allclose(
            self._run(kernel_type, events, chunks, bin_duration),
            self._run(kernel_type, events, [200], bin_duration),
            rtol=1e-9,
            atol=1e-12,
        )


class TestKernelMean(TestKernelSum):
    """MEAN with a kernel is the per-sample activation averaged over each bin's samples."""

    AGGREGATION = BinAggregation.MEAN

    @pytest.mark.parametrize("kernel_type", [ActivationKernelType.EXPONENTIAL, ActivationKernelType.ALPHA])
    @pytest.mark.parametrize("scale_by_value", [False, True], ids=["count", "weighted"])
    @pytest.mark.parametrize("bin_duration", [0.010, 0.0075], ids=["whole", "fractional"])
    def test_matches_dense_reference(self, kernel_type, scale_by_value, bin_duration):
        rng = np.random.default_rng(7)
        events = (rng.random((200, 2)) < 0.05).astype(float)
        if scale_by_value:
            events *= rng.normal(size=events.shape)
        ref = _kernel_reference(events, kernel_type, 0.008, self.FS, normalize=True)
        # Fractional 7.5-sample bins end at floor(7.5 * k): widths alternate 7 and 8.
        ends = np.floor(np.arange(1, 200 / (bin_duration * self.FS) + 1) * bin_duration * self.FS + 1e-9).astype(int)
        ends = ends[ends <= 200]
        assert set(np.diff(ends)) == ({10} if bin_duration == 0.010 else {7, 8})
        starts = np.concatenate(([0], ends[:-1]))
        expected = np.stack([ref[a:b].mean(axis=0) for a, b in zip(starts, ends)])
        out = self._run(kernel_type, events, [200], bin_duration, scale_by_value=scale_by_value)
        np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("kernel_type", [ActivationKernelType.EXPONENTIAL, ActivationKernelType.ALPHA])
    def test_irregular_chunks_match_dense_reference(self, kernel_type):
        events = (np.random.default_rng(8).random((200, 2)) < 0.05).astype(float)
        ref = _kernel_reference(events, kernel_type, 0.008, self.FS, normalize=True)
        out = self._run(kernel_type, events, [3, 1, 4, 1, 5, 9, 2, 6, 25, 3, 50, 7, 1, 83])
        np.testing.assert_allclose(out, ref.reshape(-1, 10, 2).mean(axis=1), rtol=1e-9, atol=1e-12)
