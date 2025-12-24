import numpy as np
import numpy.typing as npt


def generate_events(
    fs: float,
    dur: float,
    n_chans: int,
    rate_range: tuple[float, float],
    chunk_dur: float,
) -> list[npt.NDArray]:
    n_times = int(fs * dur)
    frates = np.random.uniform(rate_range[0], rate_range[1], n_chans)
    frates[:3] = np.random.uniform(150, 200, 3)  # Boost rate of first 3 chans.
    chunk_len = int(fs * chunk_dur)

    # Create a list of spike times for each channel
    rng = np.random.default_rng()
    spike_offsets = []
    for ch_ix, fr in enumerate(frates):
        lam, size = fs / fr, int(fr * dur)
        isi = rng.poisson(lam=lam, size=size)
        spike_samp_inds = np.cumsum(isi)
        spike_samp_inds = spike_samp_inds[spike_samp_inds < n_times]

        # Add some special cases
        if ch_ix == 0:
            # -- Refractory within chunk --
            # In channel 0, we replace the first event with a triplet; events 2-3 will be eliminated by refractory check
            spike_samp_inds = spike_samp_inds[spike_samp_inds > 30]
            spike_samp_inds = np.hstack(([1, 4, 6], spike_samp_inds))
        elif ch_ix in [1, 2]:
            # -- Unfinished events at chunk boundaries --
            # Drop spike samples within 34 samples of the end of the 0th chunk
            b_drop = np.logical_and(spike_samp_inds >= chunk_len - 34, spike_samp_inds < chunk_len)
            spike_samp_inds = spike_samp_inds[~b_drop]
            if ch_ix == 1:
                # In channel 1, we add a spike that is in the very last sample of the 0th chunk.
                # It will be detected while processing the 1th chunk.
                spike_samp_inds = np.insert(
                    spike_samp_inds,
                    np.searchsorted(spike_samp_inds, chunk_len),
                    chunk_len - 1,
                )
            elif ch_ix == 2:
                # In channel 2, we make a long event at the end of the 0th chunk.
                # It will be detected while processing the 1th chunk.
                spike_samp_inds = np.insert(
                    spike_samp_inds,
                    np.searchsorted(spike_samp_inds, chunk_len - 10),
                    np.arange(chunk_len - 10, chunk_len),
                )
        elif ch_ix == 3:
            # -- Refractoriness across chunk boundaries --
            # In channel 3, we add a spike 2 samples before the end of 1th chunk, and another within its
            #  refractory period at the beginning of 2th chunk.
            ins_ev_start = 2 * chunk_len - 2
            # Clear events that are within target period.
            b_drop = np.logical_and(
                spike_samp_inds >= ins_ev_start - 30,
                spike_samp_inds < ins_ev_start + 30,
            )
            spike_samp_inds = spike_samp_inds[~b_drop]
            # Add the two events; one 2 samples before the chunk boundary and another 10 samples later.
            spike_samp_inds = np.insert(
                spike_samp_inds,
                np.searchsorted(spike_samp_inds, ins_ev_start),
                [ins_ev_start, ins_ev_start + 10],
            )
            # Note: We must also drop events in other channels near the end of chunk 2 to make sure
            #  they don't cause the event in channel 3 to be held back to the next iteration.
        elif ch_ix == 4:
            # -- Spike in first sample of non-first chunk --
            # In channel 4, we add a spike at the very beginning of chunk 1th chunk after making sure 0th was empty.
            spike_samp_inds = spike_samp_inds[spike_samp_inds > chunk_len]
            spike_samp_inds = np.insert(spike_samp_inds, np.searchsorted(spike_samp_inds, chunk_len), chunk_len)
        spike_offsets.append(spike_samp_inds)

    # Clear all events that occur in 4th - 5th chunks to test flow logic.
    # Additionally clear events in the last sample so we don't have lingering events.
    for ch_ix, so_arr in enumerate(spike_offsets):
        b_drop = np.logical_and(so_arr >= chunk_len * 3, so_arr < chunk_len * 5)
        b_drop = np.logical_or(b_drop, so_arr == n_times - 1)
        if ch_ix != 3:
            # See above for special case in channel 3
            b_drop = np.logical_or(
                b_drop,
                np.logical_and(so_arr >= 2 * chunk_len - 30, so_arr < 2 * chunk_len),
            )
        spike_offsets[ch_ix] = so_arr[~b_drop]

    return spike_offsets


def generate_white_noise_with_events(
    fs: float,
    dur: float,
    n_chans: int,
    rate_range: tuple[float, float],
    chunk_dur: float,
    threshold: float,
) -> npt.NDArray:
    n_times = int(fs * dur)
    spike_offsets = generate_events(fs, dur, n_chans, rate_range, chunk_dur)

    rng = np.random.default_rng()
    mixed = rng.normal(size=(n_times, n_chans), loc=0, scale=0.1)
    mixed = np.clip(mixed, -np.abs(threshold), np.abs(threshold))
    for ch_ix, ch_spk_offs in enumerate(spike_offsets):
        mixed[ch_spk_offs, ch_ix] = threshold + np.random.random(size=(len(ch_spk_offs),))
    return mixed
