import numpy as np
from ezmsg.util.messages.chunker import array_chunker
import pytest

from ezmsg.event.peak import threshold_crossing


@pytest.mark.parametrize("return_peak_val", [True, False])
def test_threshold_crossing(return_peak_val: bool):
    fs = 30_000.0
    dur = 10.0
    n_chans = 128
    threshold = 2.5
    rate_range = (1, 100)
    chunk_dur = 0.02
    refrac_dur = 0.001
    n_times = int(fs * dur)
    refrac_width = int(fs * refrac_dur)
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
            b_drop = np.logical_and(
                spike_samp_inds >= chunk_len - 34, spike_samp_inds < chunk_len
            )
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
            # -- Refractory across chunk boundaries --
            # In channel 3, we add a spike 2 samples before the end of 1th chunk, and another within its
            #  refractory period at the beginning of 2th chunk.
            ins_ev_start = 2 * chunk_len - 2
            # Clear events that are within target period.
            b_drop = np.logical_and(
                spike_samp_inds >= ins_ev_start - 30,
                spike_samp_inds < ins_ev_start + 30,
            )
            spike_samp_inds = spike_samp_inds[~b_drop]
            spike_samp_inds = np.insert(
                spike_samp_inds,
                np.searchsorted(spike_samp_inds, ins_ev_start),
                [ins_ev_start, ins_ev_start + 10],
            )
            # Note: Further down we also drop events in other channels near the end of chunk 2 to make sure
            #  they don't cause the event in channel 3 to be held back to the next iteration.
        elif ch_ix == 4:
            # -- Spike in first sample of non-first chunk --
            # In channel 4, we add a spike at the very beginning of chunk 1th chunk after making sure 0th was empty.
            spike_samp_inds = spike_samp_inds[spike_samp_inds > chunk_len]
            spike_samp_inds = np.insert(
                spike_samp_inds, np.searchsorted(spike_samp_inds, chunk_len), chunk_len
            )
        spike_offsets.append(spike_samp_inds)

    # Clear all spikes that occur in 4th - 5th chunks to test flow logic.
    for ch_ix, so_arr in enumerate(spike_offsets):
        b_drop = np.logical_and(so_arr >= chunk_len * 3, so_arr < chunk_len * 5)
        if ch_ix != 3:  # See above for special case in channel 3
            b_drop = np.logical_or(
                b_drop,
                np.logical_and(so_arr >= 2 * chunk_len - 30, so_arr < 2 * chunk_len),
            )
        spike_offsets[ch_ix] = so_arr[~b_drop]

    # Create simulated spiking data: white noise + spikes
    in_dat = rng.normal(size=(n_times, n_chans), loc=0, scale=0.1)
    in_dat = np.clip(in_dat, -threshold, threshold)
    for ch_ix, ch_spk_offs in enumerate(spike_offsets):
        in_dat[ch_spk_offs, ch_ix] = threshold + np.random.random(
            size=(len(ch_spk_offs),)
        )

    bkup_dat = in_dat.copy()
    msg_gen = array_chunker(data=in_dat, chunk_len=chunk_len, axis=0, fs=fs, tzero=0.0)

    # Extract spikes
    transform = threshold_crossing(
        threshold=threshold,
        refrac_dur=refrac_dur,
        return_peak_val=return_peak_val,
    )
    msgs_out = [transform.send(_) for _ in msg_gen]

    # Calculated expected spikes -- easy to do all at once without chunk boundaries or performance constraints.
    expected = np.logical_and(bkup_dat[:-1] < threshold, bkup_dat[1:] >= threshold)
    expected = np.concatenate((np.zeros((1, n_chans), dtype=bool), expected), axis=0)
    exp_samp_inds = []
    exp_feat_inds = []
    # Remove refractory violations from expected
    for ch_ix, exp in enumerate(expected.T):
        ev_ix = np.where(exp)[0]
        while np.any(np.diff(ev_ix) <= refrac_width):
            ieis = np.hstack(([refrac_width + 1], np.diff(ev_ix)))
            drop_idx = np.where(ieis <= refrac_width)[0][0]
            ev_ix = np.delete(ev_ix, drop_idx)
        exp_samp_inds.extend(ev_ix)
        exp_feat_inds.extend([ch_ix] * len(ev_ix))

    import scipy.sparse

    final_arr = scipy.sparse.hstack([_.data for _ in msgs_out])
    feat_inds, samp_inds = final_arr.nonzero()
    assert len(samp_inds) == len(exp_samp_inds)
    assert len(feat_inds) == len(exp_feat_inds)
    assert np.array_equal(np.sort(samp_inds), np.sort(exp_samp_inds))
    assert np.array_equal(
        feat_inds[np.argsort(samp_inds)],
        np.array(exp_feat_inds)[np.argsort(exp_samp_inds)],
    )
