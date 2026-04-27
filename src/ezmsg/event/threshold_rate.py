"""Dense fused threshold-crossing event-rate transformer."""

from typing import Any

import ezmsg.core as ez
import numpy as np
from array_api_compat import get_namespace, is_numpy_array
from ezmsg.baseproc import BaseStatefulTransformer, BaseTransformerUnit, processor_state
from ezmsg.util.messages.axisarray import AxisArray, replace


class ThresholdCrossingRateSettings(ez.Settings):
    threshold: float = -3.5
    """The value the signal must cross to count an event."""

    refrac_dur: float = 0.001
    """Minimum duration between counted threshold crossings in seconds."""

    bin_duration: float = 0.05
    """Output bin duration in seconds."""

    rate_normalize: bool = True
    """If True, divide counts by bin_duration to emit events/second."""

    axis: str = "time"
    """Input sample axis."""

    use_mlx_metal: bool = True
    """If True, MLX inputs use the fused on-device Metal implementation."""


@processor_state
class ThresholdCrossingRateState:
    prev_over: Any = None
    """Whether the previous sample was over threshold for each feature."""

    elapsed: Any = None
    """Samples since the last accepted threshold crossing for each feature."""

    overflow_counts: Any = None
    """Raw counts in the current partial output bin for each feature."""

    n_overflow: int = 0
    """Number of input samples in the current partial output bin."""

    refrac_width: int = 0
    samples_per_bin: int = 0


class ThresholdCrossingRateTransformer(
    BaseStatefulTransformer[
        ThresholdCrossingRateSettings,
        AxisArray,
        AxisArray,
        ThresholdCrossingRateState,
    ]
):
    """Count threshold crossings directly into dense rate bins.

    This transformer covers the simple threshold-crossing case used by dense
    preprocessing pipelines: crossing-aligned events, no peak-value payload,
    no peak-duration filtering, and no sparse.COO boundary. It preserves exact
    refractory behavior while allowing MLX inputs to remain on device through
    a fused Metal path.
    """

    def _hash_message(self, message: AxisArray) -> int:
        ax_idx = message.get_axis_idx(self.settings.axis)
        sample_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1 :]
        return hash((message.key, sample_shape, message.axes[self.settings.axis].gain))

    def _reset_state(self, message: AxisArray) -> None:
        xp = get_namespace(message.data)
        ax_idx = message.get_axis_idx(self.settings.axis)
        feature_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1 :]

        fs = 1.0 / message.axes[self.settings.axis].gain
        self._state.refrac_width = int(self.settings.refrac_dur * fs)
        self._state.samples_per_bin = int(self.settings.bin_duration * fs)
        if self._state.samples_per_bin < 1:
            raise ValueError(
                f"bin_duration={self.settings.bin_duration} is shorter than one sample at fs={fs:g} Hz"
            )

        self._state.prev_over = None
        self._state.elapsed = xp.full(
            feature_shape,
            self._state.refrac_width + 1,
            dtype=xp.int32,
        )
        self._state.overflow_counts = xp.zeros(feature_shape, dtype=xp.float32)
        self._state.n_overflow = 0

    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)
        ax_idx = message.get_axis_idx(self.settings.axis)

        if ax_idx != 0:
            perm = (ax_idx,) + tuple(i for i in range(message.data.ndim) if i != ax_idx)
            message = replace(
                message,
                data=xp.permute_dims(message.data, perm),
                dims=[self.settings.axis] + message.dims[:ax_idx] + message.dims[ax_idx + 1 :],
            )

        n_samples = message.data.shape[0]
        n_prev_overflow = self._state.n_overflow
        n_total = n_prev_overflow + n_samples
        n_bins = n_total // self._state.samples_per_bin
        self._state.n_overflow = n_total - n_bins * self._state.samples_per_bin

        if n_samples == 0:
            feature_shape = message.data.shape[1:]
            out_data = xp.zeros((n_bins,) + feature_shape, dtype=xp.float32)
        elif (
            self.settings.use_mlx_metal
            and not is_numpy_array(message.data)
            and getattr(xp, "__name__", "") == "mlx.core"
        ):
            out_data = self._process_mlx(message.data, n_prev_overflow, n_bins)
        else:
            out_data = self._process_numpy(message.data, n_prev_overflow, n_bins)

        time_axis = message.axes[self.settings.axis]
        out_offset = time_axis.offset if n_bins == 0 else time_axis.offset - n_prev_overflow * time_axis.gain
        out_axis = replace(
            time_axis,
            gain=self.settings.bin_duration,
            offset=out_offset,
        )
        return replace(
            message,
            data=out_data,
            axes={**message.axes, self.settings.axis: out_axis},
        )

    def _process_numpy(self, data, n_prev_overflow: int, n_bins: int):
        data_np = data if is_numpy_array(data) else np.asarray(data)
        n_samples = data_np.shape[0]
        feature_shape = data_np.shape[1:]
        n_features = int(np.prod(feature_shape, dtype=np.int64)) if feature_shape else 1
        flat = data_np.reshape(n_samples, n_features)

        prev_over = self._state.prev_over
        if prev_over is None:
            prev_flat = _initial_prev_over(flat, self.settings.threshold)
        else:
            prev_flat = np.asarray(prev_over, dtype=bool).reshape(n_features)

        elapsed_flat = np.asarray(self._state.elapsed, dtype=np.int32).reshape(n_features).copy()
        overflow_flat = np.asarray(self._state.overflow_counts, dtype=np.float32).reshape(n_features).copy()

        out = np.zeros((n_bins, n_features), dtype=np.float32)
        if n_bins > 0:
            out[0] = overflow_flat
            overflow_flat.fill(0.0)

        for samp_ix in range(n_samples):
            sample = flat[samp_ix]
            if self.settings.threshold >= 0:
                over = sample >= self.settings.threshold
            else:
                over = sample <= self.settings.threshold
            crossing = over & ~prev_flat
            prev_flat = over

            elapsed_flat += 1
            if self._state.refrac_width > 2:
                accepted = crossing & (elapsed_flat > self._state.refrac_width)
            else:
                accepted = crossing
            if np.any(accepted):
                bin_ix = (n_prev_overflow + samp_ix) // self._state.samples_per_bin
                accepted_f32 = accepted.astype(np.float32)
                if bin_ix < n_bins:
                    out[bin_ix] += accepted_f32
                else:
                    overflow_flat += accepted_f32
                elapsed_flat[accepted] = 0

        if self.settings.rate_normalize:
            out /= self.settings.bin_duration

        self._state.prev_over = prev_flat.reshape(feature_shape)
        self._state.elapsed = elapsed_flat.reshape(feature_shape)
        self._state.overflow_counts = overflow_flat.reshape(feature_shape)
        return out.reshape((n_bins,) + feature_shape)

    def _process_mlx(self, data, n_prev_overflow: int, n_bins: int):
        import mlx.core as mx

        from ezmsg.event.util.threshold_rate_mlx_metal import threshold_crossing_rate_mlx_metal

        if self._state.prev_over is None:
            first = data[0]
            if self.settings.threshold >= 0:
                over = first >= self.settings.threshold
            else:
                over = first <= self.settings.threshold
            self._state.prev_over = over.astype(mx.uint32)

        self._state.elapsed = mx.asarray(self._state.elapsed, dtype=mx.int32)
        self._state.overflow_counts = mx.asarray(self._state.overflow_counts, dtype=mx.float32)
        self._state.prev_over = mx.asarray(self._state.prev_over, dtype=mx.uint32)

        out, self._state.prev_over, self._state.elapsed, self._state.overflow_counts = (
            threshold_crossing_rate_mlx_metal(
                data,
                self._state.prev_over,
                self._state.elapsed,
                self._state.overflow_counts,
                threshold=self.settings.threshold,
                refrac_width=self._state.refrac_width,
                n_overflow=n_prev_overflow,
                samples_per_bin=self._state.samples_per_bin,
                n_bins=n_bins,
                bin_duration=self.settings.bin_duration,
                rate_normalize=self.settings.rate_normalize,
            )
        )
        return out


def _initial_prev_over(flat: np.ndarray, threshold: float) -> np.ndarray:
    n_features = flat.shape[1]
    if flat.shape[0] == 0:
        return np.zeros(n_features, dtype=bool)
    first = flat[0]
    return first >= threshold if threshold >= 0 else first <= threshold


class ThresholdCrossingRate(
    BaseTransformerUnit[
        ThresholdCrossingRateSettings,
        AxisArray,
        AxisArray,
        ThresholdCrossingRateTransformer,
    ]
):
    """Unit for dense threshold-crossing event rates."""

    SETTINGS = ThresholdCrossingRateSettings
