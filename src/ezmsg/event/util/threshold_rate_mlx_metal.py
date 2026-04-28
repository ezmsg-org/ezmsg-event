"""Fused threshold-crossing rate calculation on Apple Silicon via MLX + Metal."""

import mlx.core as mx


def threshold_crossing_rate_mlx_metal(
    x,
    prev_over,
    elapsed,
    overflow_counts,
    *,
    threshold: float,
    refrac_width: int,
    bin_accumulator: float,
    samples_per_bin: float,
    n_bins: int,
    bin_duration: float,
    rate_normalize: bool,
):
    """Compute threshold-crossing rates for ``x`` with time on axis 0.

    Args:
        x: MLX array with shape ``(n_samples, *features)``.
        prev_over: Int8 MLX array with shape ``(*features,)``; nonzero
            indicates whether the sample before this chunk was over threshold.
        elapsed: Int32 MLX array with shape ``(*features,)`` tracking samples
            since the last accepted crossing.
        overflow_counts: Float32 MLX array with shape ``(*features,)`` holding
            raw counts in the partial output bin carried from previous chunks.
        threshold: Threshold crossing level.
        refrac_width: Refractory duration in samples. A crossing is accepted
            only when the distance from the previous accepted crossing is
            greater than this value.
        bin_accumulator: Fractional number of input samples already accumulated
            in the current partial output bin.
        samples_per_bin: Fractional number of input samples per output bin.
        n_bins: Number of complete output bins produced by this chunk.
        bin_duration: Output bin duration in seconds.
        rate_normalize: If true, output events/second; otherwise raw counts.

    Returns:
        ``(rates, prev_over, elapsed, overflow_counts)``. ``rates`` has shape
        ``(n_bins, *features)`` and the state outputs are shaped like the state
        inputs.
    """
    if x.ndim < 1:
        raise ValueError(f"x must have at least 1 dimension; got {x.ndim}")
    if samples_per_bin < 1.0:
        raise ValueError(f"samples_per_bin must be >= 1.0; got {samples_per_bin}")
    if n_bins < 0:
        raise ValueError(f"n_bins must be >= 0; got {n_bins}")

    x_f32 = x.astype(mx.float32) if x.dtype != mx.float32 else x
    batch_shape = tuple(x_f32.shape[1:])
    n_samples = x_f32.shape[0]
    n_channels = 1
    for dim in batch_shape:
        n_channels *= dim

    if n_samples == 0:
        rates = mx.zeros((n_bins,) + batch_shape, dtype=mx.float32)
        return (
            rates,
            prev_over.astype(mx.int8).reshape(batch_shape),
            elapsed.astype(mx.int32).reshape(batch_shape),
            overflow_counts.astype(mx.float32).reshape(batch_shape),
        )

    x_flat = x_f32.reshape(n_samples, n_channels)
    prev_flat = prev_over.astype(mx.int8).reshape(n_channels)
    elapsed_flat = elapsed.astype(mx.int32).reshape(n_channels)
    overflow_flat = overflow_counts.astype(mx.float32).reshape(n_channels)
    n_words = (n_samples + 31) // 32

    params = mx.array(
        [
            float(threshold),
            float(1.0 / bin_duration if rate_normalize else 1.0),
            float(samples_per_bin - bin_accumulator),
            float(samples_per_bin),
        ],
        dtype=mx.float32,
    )

    crossing_words, final_over = _crossing_words_kernel(
        inputs=[x_flat, prev_flat, params],
        template=[
            ("N_SAMPLES", n_samples),
            ("N_CHANNELS", n_channels),
            ("N_WORDS", n_words),
        ],
        grid=(n_words, n_channels, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(n_words, n_channels), (n_channels,)],
        output_dtypes=[mx.uint32, mx.int8],
    )

    # Metal kernels cannot emit a zero-size output on all MLX versions. Use a
    # one-bin scratch output and slice it away for chunks with no complete bin.
    n_output_bins = max(n_bins, 1)
    rates_flat, prev_out, elapsed_out, overflow_out = _refractory_words_kernel(
        inputs=[crossing_words, final_over, elapsed_flat, overflow_flat, params],
        template=[
            ("N_SAMPLES", n_samples),
            ("N_CHANNELS", n_channels),
            ("N_WORDS", n_words),
            ("N_BINS", n_bins),
            ("N_OUTPUT_BINS", n_output_bins),
            ("REFRAC_WIDTH", refrac_width),
        ],
        grid=(n_channels, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[
            (n_output_bins, n_channels),
            (n_channels,),
            (n_channels,),
            (n_channels,),
        ],
        output_dtypes=[mx.float32, mx.int8, mx.int32, mx.float32],
    )

    rates_flat = rates_flat[:n_bins]
    rates = rates_flat.reshape((n_bins,) + batch_shape)
    return (
        rates,
        prev_out.reshape(batch_shape),
        elapsed_out.reshape(batch_shape),
        overflow_out.reshape(batch_shape),
    )


_CROSSING_WORDS_KERNEL_SOURCE = r"""
    uint word = thread_position_in_grid.x;
    uint ch = thread_position_in_grid.y;
    if (word >= N_WORDS || ch >= N_CHANNELS) {
        return;
    }

    float threshold = params[0];
    uint start = word * 32;
    uint prev = 0;
    if (start == 0) {
        prev = prev_over_in[ch] != 0;
    } else {
        float prev_sample = x_in[(start - 1) * N_CHANNELS + ch];
        prev = threshold >= 0.0f ? (prev_sample >= threshold) : (prev_sample <= threshold);
    }

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
        final_over_out[ch] = prev ? 1 : 0;
    }
"""


_REFRACTORY_WORDS_KERNEL_SOURCE = r"""
    uint ch = thread_position_in_grid.x;
    if (ch >= N_CHANNELS) {
        return;
    }

    float scale = params[1];
    float first_bin_end = params[2];
    float samples_per_bin = params[3];

    float overflow = overflow_counts_in[ch];
    for (uint bin = 0; bin < N_OUTPUT_BINS; ++bin) {
        rates_out[bin * N_CHANNELS + ch] = 0.0f;
    }
    if (N_BINS > 0) {
        rates_out[ch] = overflow;
        overflow = 0.0f;
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
                int active_bin = int(ceil((float(t) + 1.0f - first_bin_end) / samples_per_bin));
                if (active_bin >= 0 && active_bin < N_BINS) {
                    rates_out[uint(active_bin) * N_CHANNELS + ch] += 1.0f;
                } else {
                    overflow += 1.0f;
                }
                elapsed = 0;
            }
        }
    }

    elapsed += int(N_SAMPLES) - 1 - last_t;

    for (uint bin = 0; bin < N_BINS; ++bin) {
        rates_out[bin * N_CHANNELS + ch] *= scale;
    }

    prev_over_out[ch] = final_over_in[ch];
    elapsed_out[ch] = elapsed;
    overflow_counts_out[ch] = overflow;
"""


_crossing_words_kernel = mx.fast.metal_kernel(
    name="threshold_crossing_words",
    input_names=["x_in", "prev_over_in", "params"],
    output_names=["crossing_words_out", "final_over_out"],
    source=_CROSSING_WORDS_KERNEL_SOURCE,
)


_refractory_words_kernel = mx.fast.metal_kernel(
    name="threshold_refractory_words",
    input_names=[
        "crossing_words_in",
        "final_over_in",
        "elapsed_in",
        "overflow_counts_in",
        "params",
    ],
    output_names=["rates_out", "prev_over_out", "elapsed_out", "overflow_counts_out"],
    source=_REFRACTORY_WORDS_KERNEL_SOURCE,
)
