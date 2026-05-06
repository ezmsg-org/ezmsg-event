"""On-device threshold-crossing detection on Apple Silicon via MLX + Metal.

Two fused metal kernels handle the parts of threshold detection that are
sequential on the time axis (and thus hard to vectorize cleanly): finding
threshold crossings, then enforcing the refractory period. The output is a
dense events array in the input's namespace, so downstream nodes (e.g.
:class:`ezmsg.event.kernel_activation.BinnedKernelActivation`) can keep the
data on the GPU.

Adapted from a fused threshold-rate kernel originally written by kylmcgr:
the bin-sum stage was moved out of the kernel and into
``BinnedKernelActivation`` so that ``ThresholdCrossingTransformer`` produces
the same dense events tensor regardless of backend.
"""

import mlx.core as mx


def threshold_crossings_mlx_metal(
    x,
    prev_sample,
    elapsed,
    *,
    threshold: float,
    refrac_width: int,
):
    """Detect threshold crossings + enforce refractory; emit dense events.

    Args:
        x: MLX array with shape ``(n_samples, *features)``.
        prev_sample: Float32 MLX array shaped ``(*features,)`` — the sample
            value immediately preceding this chunk (used as the over/under
            reference for the first crossing comparison). On the very first
            chunk, pass the chunk's own first sample so it is never reported
            as a crossing.
        elapsed: Int32 MLX array shaped ``(*features,)`` tracking samples
            since the last accepted crossing (initialise to ``refrac_width + 1``
            so the first sample is eligible).
        threshold: Crossing level. Values ``>= 0`` look for upward crossings
            (signal must rise to or above ``threshold`` from below); values
            ``< 0`` look for downward crossings.
        refrac_width: Refractory duration in samples. A crossing is accepted
            only when it is more than ``refrac_width`` samples past the
            previous accepted crossing. Values ``<= 2`` disable the gate.

    Returns:
        ``(events, last_sample, elapsed_out)`` where ``events`` is an int8
        MLX array shaped ``(n_samples, *features)`` with 1 at each accepted
        crossing (0 elsewhere). ``last_sample`` is float32 — the chunk's final
        sample, ready to be fed back as ``prev_sample`` next call. The state
        outputs are shaped like the state inputs.
    """
    if x.ndim < 1:
        raise ValueError(f"x must have at least 1 dimension; got {x.ndim}")

    x_f32 = x.astype(mx.float32) if x.dtype != mx.float32 else x
    batch_shape = tuple(x_f32.shape[1:])
    n_samples = x_f32.shape[0]
    n_channels = 1
    for dim in batch_shape:
        n_channels *= dim

    if n_samples == 0:
        events = mx.zeros((0,) + batch_shape, dtype=mx.int8)
        return (
            events,
            prev_sample.astype(mx.float32).reshape(batch_shape),
            elapsed.astype(mx.int32).reshape(batch_shape),
        )

    x_flat = x_f32.reshape(n_samples, n_channels)
    prev_sample_flat = prev_sample.astype(mx.float32).reshape(n_channels)
    elapsed_flat = elapsed.astype(mx.int32).reshape(n_channels)
    n_words = (n_samples + 31) // 32

    params = mx.array([float(threshold)], dtype=mx.float32)

    crossing_words, last_sample = _crossing_words_kernel(
        inputs=[x_flat, prev_sample_flat, params],
        template=[
            ("N_SAMPLES", n_samples),
            ("N_CHANNELS", n_channels),
            ("N_WORDS", n_words),
        ],
        grid=(n_words, n_channels, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(n_words, n_channels), (n_channels,)],
        output_dtypes=[mx.uint32, mx.float32],
    )

    events_flat, elapsed_out = _refractory_dense_kernel(
        inputs=[crossing_words, elapsed_flat],
        template=[
            ("N_SAMPLES", n_samples),
            ("N_CHANNELS", n_channels),
            ("N_WORDS", n_words),
            ("REFRAC_WIDTH", refrac_width),
        ],
        grid=(n_channels, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(n_samples, n_channels), (n_channels,)],
        output_dtypes=[mx.int8, mx.int32],
    )

    events = events_flat.reshape((n_samples,) + batch_shape)
    return (
        events,
        last_sample.reshape(batch_shape),
        elapsed_out.reshape(batch_shape),
    )


_CROSSING_WORDS_KERNEL_SOURCE = r"""
    uint word = thread_position_in_grid.x;
    uint ch = thread_position_in_grid.y;
    if (word >= N_WORDS || ch >= N_CHANNELS) {
        return;
    }

    float threshold = params[0];
    uint start = word * 32;
    float ref_sample = start == 0 ? prev_sample_in[ch] : x_in[(start - 1) * N_CHANNELS + ch];
    uint prev = threshold >= 0.0f ? (ref_sample >= threshold) : (ref_sample <= threshold);

    uint bits = 0;
    for (uint bit = 0; bit < 32; ++bit) {
        uint t = start + bit;
        if (t >= N_SAMPLES) {
            break;
        }

        float sample = x_in[t * N_CHANNELS + ch];
        uint over = threshold >= 0.0f ? (sample >= threshold) : (sample <= threshold);
        if (over && !prev) {
            bits |= (1u << bit);
        }
        prev = over;
    }
    crossing_words_out[word * N_CHANNELS + ch] = bits;

    if (word == N_WORDS - 1) {
        last_sample_out[ch] = x_in[(N_SAMPLES - 1) * N_CHANNELS + ch];
    }
"""


_REFRACTORY_DENSE_KERNEL_SOURCE = r"""
    uint ch = thread_position_in_grid.x;
    if (ch >= N_CHANNELS) {
        return;
    }

    for (uint t = 0; t < N_SAMPLES; ++t) {
        events_out[t * N_CHANNELS + ch] = 0;
    }

    int elapsed = elapsed_in[ch];
    int last_t = -1;

    for (uint word = 0; word < N_WORDS; ++word) {
        uint bits = crossing_words_in[word * N_CHANNELS + ch];
        while (bits != 0) {
            uint bit = 0;
            uint mask = 1u;
            while ((bits & mask) == 0u) {
                bit += 1;
                mask <<= 1;
            }
            bits &= ~mask;

            uint t = word * 32 + bit;
            if (t >= N_SAMPLES) {
                break;
            }

            elapsed += int(t) - last_t;
            last_t = int(t);

            if (REFRAC_WIDTH <= 2 || elapsed > REFRAC_WIDTH) {
                events_out[t * N_CHANNELS + ch] = 1;
                elapsed = 0;
            }
        }
    }

    elapsed += int(N_SAMPLES) - 1 - last_t;
    elapsed_out[ch] = elapsed;
"""


_crossing_words_kernel = mx.fast.metal_kernel(
    name="threshold_crossing_words",
    input_names=["x_in", "prev_sample_in", "params"],
    output_names=["crossing_words_out", "last_sample_out"],
    source=_CROSSING_WORDS_KERNEL_SOURCE,
)


_refractory_dense_kernel = mx.fast.metal_kernel(
    name="threshold_refractory_dense",
    input_names=["crossing_words_in", "elapsed_in"],
    output_names=["events_out", "elapsed_out"],
    source=_REFRACTORY_DENSE_KERNEL_SOURCE,
)
