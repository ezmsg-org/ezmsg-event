import cProfile

# import pstats
import numpy as np
import sparse
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.event.rate import EventRate, EventRateSettings


# Simulate input data
def generate_sparse_data(num_samples, num_channels, sparsity_factor, rng):
    data = sparse.random((num_samples, num_channels), density=sparsity_factor, random_state=rng) > 0
    return data


def run_rate_processor(num_samples, num_channels, sparsity_factor, bin_duration, chunk_dur, fs):
    rng = np.random.default_rng()
    s = generate_sparse_data(num_samples, num_channels, sparsity_factor, rng)

    chunk_len = int(fs * chunk_dur)
    nchunk = int(num_samples / chunk_len)

    in_msgs = [
        AxisArray(
            data=s[chunk_ix * chunk_len : (chunk_ix + 1) * chunk_len],
            dims=["time", "ch"],
            axes={
                "time": AxisArray.Axis.TimeAxis(fs=fs, offset=chunk_ix * chunk_dur),
            },
        )
        for chunk_ix in range(nchunk)
    ]

    settings = EventRateSettings(bin_duration=bin_duration)
    rate_processor = EventRate(settings)

    output_messages = []
    for in_msg in in_msgs:
        output_messages.append(rate_processor(in_msg))

    return output_messages


if __name__ == "__main__":
    NUM_SAMPLES = 300_000  # Number of time samples (e.g., 10 seconds at 30kHz)
    NUM_CHANNELS = 128  # Number of channels
    SPARSITY_FACTOR = 0.0001  # 0.01% sparse, as in test_rate.py
    BIN_DURATION = 0.03  # 30ms bin duration, as in test_rate.py
    CHUNK_DURATION = 0.1  # 100ms chunk duration, as in test_rate.py
    FS = 30000.0  # Sampling frequency, as in test_rate.py

    print(
        f"Profiling with: samples={NUM_SAMPLES}, channels={NUM_CHANNELS}, sparsity={SPARSITY_FACTOR},"
        f"bin_duration={BIN_DURATION}, chunk_duration={CHUNK_DURATION}, fs={FS}"
    )

    # Run with cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    run_rate_processor(NUM_SAMPLES, NUM_CHANNELS, SPARSITY_FACTOR, BIN_DURATION, CHUNK_DURATION, FS)

    profiler.disable()

    # Save stats to a file in a format snakeviz can read
    stats_file = "rate_profile.prof"
    profiler.dump_stats(stats_file)

    print(f"Profiling results saved to {stats_file}")
    print("To visualize, run: uv run snakeviz rate_profile.prof")
