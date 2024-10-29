import numpy as np
from ezmsg.util.messages.chunker import array_chunker
import pytest

from ezmsg.event.util.simulate import generate_events
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
    chunk_len = int(fs * chunk_dur)

    spike_offsets = generate_events(fs, dur, n_chans, rate_range, chunk_dur)

    # Create simulated spiking data: white noise + spikes
    rng = np.random.default_rng()
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
    exp_feat_inds = np.array(exp_feat_inds)
    exp_samp_inds = np.array(exp_samp_inds)

    import scipy.sparse

    final_arr = scipy.sparse.hstack([_.data for _ in msgs_out])
    feat_inds, samp_inds = final_arr.nonzero()

    """
    # This block of code was used to debug some discrepancies that popped up when the last sample of the last chunk
    #  had an event, but the processing node wouldn't return it because it was unfinished.
    if len(samp_inds) != len(exp_samp_inds):
        uq_feats, feat_splits = np.unique(feat_inds, return_index=True)
        feat_crosses = {k: v for k, v in zip(uq_feats, np.split(samp_inds, feat_splits[1:]))}
        uq_feats, feat_splits = np.unique(exp_feat_inds, return_index=True)
        exp_feat_crosses = {k: v for k, v in zip(uq_feats, np.split(exp_samp_inds, feat_splits[1:]))}
        for k, v in feat_crosses.items():
            if not np.array_equal(v, exp_feat_crosses[k]):
                print(f"Channel {k}:")
                if len(exp_feat_crosses[k]) > len(v):
                    print(f"\tMissing: {np.setdiff1d(exp_feat_crosses[k], v)}")
                else:
                    print(f"\tExtra: {np.setdiff1d(v, exp_feat_crosses[k])}")
    """

    assert len(samp_inds) == len(exp_samp_inds)
    assert len(feat_inds) == len(exp_feat_inds)
    assert np.array_equal(samp_inds, exp_samp_inds)
    assert np.array_equal(feat_inds, exp_feat_inds)
