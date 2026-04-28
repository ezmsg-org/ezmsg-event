import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.event.peak import ThresholdCrossingTransformer
from ezmsg.event.rate import EventRateSettings, Rate
from ezmsg.event.threshold_rate import ThresholdCrossingRateSettings, ThresholdCrossingRateTransformer


def _make_msg(data: np.ndarray, fs: float, offset: float, dims: list[str] | None = None) -> AxisArray:
    dims = dims or ["time", "ch"]
    return AxisArray(
        data=data,
        dims=dims,
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=offset),
            "ch": AxisArray.CoordinateAxis(
                data=np.arange(data.shape[dims.index("ch")]),
                dims=["ch"],
            ),
        },
    )


def _run_sparse_reference(
    chunks: list[np.ndarray],
    *,
    fs: float,
    threshold: float,
    refrac_dur: float,
    bin_duration: float,
    dims: list[str] | None = None,
) -> list[AxisArray]:
    thresh = ThresholdCrossingTransformer(threshold=threshold, refrac_dur=refrac_dur)
    rate = Rate(EventRateSettings(bin_duration=bin_duration))
    out = []
    samp_offset = 0
    for chunk in chunks:
        msg = _make_msg(chunk, fs, samp_offset / fs, dims=dims)
        out.append(rate(thresh(msg)))
        samp_offset += chunk.shape[dims.index("time") if dims else 0]
    return out


def _run_dense_fused(
    chunks: list[np.ndarray],
    *,
    fs: float,
    threshold: float,
    refrac_dur: float,
    bin_duration: float,
    dims: list[str] | None = None,
) -> list[AxisArray]:
    proc = ThresholdCrossingRateTransformer(
        ThresholdCrossingRateSettings(
            threshold=threshold,
            refrac_dur=refrac_dur,
            bin_duration=bin_duration,
            use_mlx_metal=False,
        )
    )
    out = []
    samp_offset = 0
    for chunk in chunks:
        msg = _make_msg(chunk, fs, samp_offset / fs, dims=dims)
        out.append(proc(msg))
        samp_offset += chunk.shape[dims.index("time") if dims else 0]
    return out


def _require_mlx_metal():
    mx = pytest.importorskip("mlx.core")
    try:
        probe = mx.array([1.0], dtype=mx.float32)
        mx.eval(probe)
    except RuntimeError as exc:
        pytest.skip(f"MLX Metal device unavailable: {exc}")
    return mx


def _assert_messages_match(actual: list[AxisArray], expected: list[AxisArray]) -> None:
    assert len(actual) == len(expected)
    for actual_msg, expected_msg in zip(actual, expected):
        assert actual_msg.dims == expected_msg.dims
        assert actual_msg.data.shape == expected_msg.data.shape
        assert actual_msg.axes["time"].gain == expected_msg.axes["time"].gain
        assert np.isclose(actual_msg.axes["time"].offset, expected_msg.axes["time"].offset)
        np.testing.assert_allclose(actual_msg.data, expected_msg.data)


def test_threshold_crossing_rate_matches_sparse_pipeline_with_refractory_and_overflow():
    fs = 1000.0
    threshold = 1.0
    refrac_dur = 0.006
    bin_duration = 0.010
    data = np.zeros((43, 3), dtype=np.float32)

    # Channel 0 exercises greedy refractory: sample 5 is dropped after sample 1,
    # but sample 9 is accepted because dropped crossings do not extend refractory.
    for samp, ch in [
        (0, 2),  # first sample has no prior reference, so it should not count
        (1, 0),
        (5, 0),
        (9, 0),
        (12, 1),
        (19, 1),
        (20, 1),
        (30, 2),
        (38, 2),
    ]:
        data[samp, ch] = 2.0

    chunks = [data[:4], data[4:12], data[12:19], data[19:]]
    expected = _run_sparse_reference(
        chunks,
        fs=fs,
        threshold=threshold,
        refrac_dur=refrac_dur,
        bin_duration=bin_duration,
    )
    actual = _run_dense_fused(
        chunks,
        fs=fs,
        threshold=threshold,
        refrac_dur=refrac_dur,
        bin_duration=bin_duration,
    )

    _assert_messages_match(actual, expected)


def test_threshold_crossing_rate_matches_sparse_pipeline_for_negative_threshold():
    fs = 1000.0
    threshold = -1.0
    refrac_dur = 0.004
    bin_duration = 0.010
    data = np.zeros((31, 2), dtype=np.float32)
    for samp, ch in [(0, 0), (3, 0), (8, 0), (12, 1), (17, 1), (24, 1)]:
        data[samp, ch] = -2.0

    chunks = [data[:6], data[6:16], data[16:]]
    expected = _run_sparse_reference(
        chunks,
        fs=fs,
        threshold=threshold,
        refrac_dur=refrac_dur,
        bin_duration=bin_duration,
    )
    actual = _run_dense_fused(
        chunks,
        fs=fs,
        threshold=threshold,
        refrac_dur=refrac_dur,
        bin_duration=bin_duration,
    )

    _assert_messages_match(actual, expected)


def test_threshold_crossing_rate_matches_sparse_pipeline_for_fractional_samples_per_bin():
    fs = 30012.0048
    threshold = 1.0
    refrac_dur = 0.0
    bin_duration = 0.020
    data = np.zeros((5000, 2), dtype=np.float32)
    for samp, ch in [
        (599, 0),
        (600, 1),
        (1199, 0),
        (1200, 1),
        (2399, 0),
        (2400, 1),
        (3000, 0),
        (3001, 1),
        (3602, 0),
    ]:
        data[samp, ch] = 2.0

    chunks = [data[:777], data[777:1310], data[1310:2450], data[2450:3800], data[3800:]]
    expected = _run_sparse_reference(
        chunks,
        fs=fs,
        threshold=threshold,
        refrac_dur=refrac_dur,
        bin_duration=bin_duration,
    )
    actual = _run_dense_fused(
        chunks,
        fs=fs,
        threshold=threshold,
        refrac_dur=refrac_dur,
        bin_duration=bin_duration,
    )

    _assert_messages_match(actual, expected)


def test_threshold_crossing_rate_supports_nonzero_time_axis():
    fs = 1000.0
    threshold = 1.0
    refrac_dur = 0.004
    bin_duration = 0.010
    data = np.zeros((2, 25), dtype=np.float32)
    data[0, [2, 8, 14]] = 2.0
    data[1, [5, 11, 21]] = 2.0

    chunks = [data[:, :9], data[:, 9:17], data[:, 17:]]
    dims = ["ch", "time"]
    expected = _run_sparse_reference(
        chunks,
        fs=fs,
        threshold=threshold,
        refrac_dur=refrac_dur,
        bin_duration=bin_duration,
        dims=dims,
    )
    actual = _run_dense_fused(
        chunks,
        fs=fs,
        threshold=threshold,
        refrac_dur=refrac_dur,
        bin_duration=bin_duration,
        dims=dims,
    )

    _assert_messages_match(actual, expected)


def test_threshold_crossing_rate_empty_time_first_and_midstream():
    fs = 1000.0
    proc = ThresholdCrossingRateTransformer(
        threshold=1.0,
        refrac_dur=0.004,
        bin_duration=0.010,
        use_mlx_metal=False,
    )

    empty = _make_msg(np.zeros((0, 2), dtype=np.float32), fs, 0.0)
    out_empty = proc(empty)
    assert out_empty.data.shape == (0, 2)

    data = np.zeros((20, 2), dtype=np.float32)
    data[0, 0] = 2.0
    data[3, 0] = 2.0
    data[11, 1] = 2.0
    out_data = proc(_make_msg(data, fs, 0.0))

    ref_thresh = ThresholdCrossingTransformer(threshold=1.0, refrac_dur=0.004)
    ref_rate = Rate(EventRateSettings(bin_duration=0.010))
    ref_rate(ref_thresh(empty))
    expected = ref_rate(ref_thresh(_make_msg(data, fs, 0.0)))
    np.testing.assert_allclose(out_data.data, expected.data)

    mid_empty = proc(_make_msg(np.zeros((0, 2), dtype=np.float32), fs, 0.020))
    assert mid_empty.data.shape == (0, 2)


def test_threshold_crossing_rate_mlx_metal_matches_cpu_for_adversarial_refractory():
    mx = _require_mlx_metal()
    fs = 1000.0
    threshold = 1.0
    refrac_dur = 0.030
    bin_duration = 0.100
    data = np.zeros((1000, 4), dtype=np.float32)

    for ch in range(data.shape[1]):
        for samp in range(ch + 1, data.shape[0], 31):
            data[samp, ch] = 2.0

    chunks = [data[:137], data[137:503], data[503:777], data[777:]]
    expected = _run_dense_fused(
        chunks,
        fs=fs,
        threshold=threshold,
        refrac_dur=refrac_dur,
        bin_duration=bin_duration,
    )

    proc = ThresholdCrossingRateTransformer(
        ThresholdCrossingRateSettings(
            threshold=threshold,
            refrac_dur=refrac_dur,
            bin_duration=bin_duration,
            use_mlx_metal=True,
        )
    )
    actual = []
    samp_offset = 0
    for chunk in chunks:
        msg = _make_msg(mx.array(chunk), fs, samp_offset / fs)
        out = proc(msg)
        mx.eval(
            out.data,
            proc._state.prev_over,
            proc._state.elapsed,
            proc._state.overflow_counts,
        )
        actual.append(out)
        samp_offset += chunk.shape[0]

    _assert_messages_match(actual, expected)


def test_threshold_crossing_rate_mlx_metal_matches_cpu_for_negative_fractional_bins():
    mx = _require_mlx_metal()
    fs = 30012.0048
    threshold = -1.0
    refrac_dur = 0.001
    bin_duration = 0.020
    data = np.zeros((5000, 3), dtype=np.float32)
    for samp, ch in [
        (599, 0),
        (600, 1),
        (631, 1),
        (1199, 0),
        (1200, 2),
        (1230, 2),
        (2399, 0),
        (2400, 1),
        (3000, 0),
        (3001, 2),
        (3602, 1),
    ]:
        data[samp, ch] = -2.0

    chunks = [data[:777], data[777:1310], data[1310:2450], data[2450:3800], data[3800:]]
    expected = _run_dense_fused(
        chunks,
        fs=fs,
        threshold=threshold,
        refrac_dur=refrac_dur,
        bin_duration=bin_duration,
    )

    proc = ThresholdCrossingRateTransformer(
        ThresholdCrossingRateSettings(
            threshold=threshold,
            refrac_dur=refrac_dur,
            bin_duration=bin_duration,
            use_mlx_metal=True,
        )
    )
    actual = []
    samp_offset = 0
    for chunk in chunks:
        msg = _make_msg(mx.array(chunk), fs, samp_offset / fs)
        out = proc(msg)
        mx.eval(
            out.data,
            proc._state.prev_over,
            proc._state.elapsed,
            proc._state.overflow_counts,
        )
        actual.append(out)
        samp_offset += chunk.shape[0]

    _assert_messages_match(actual, expected)
