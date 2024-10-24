
import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.chunker import array_chunker
from ezmsg.event.peak import threshold_crossing


def test_threshold_crossing():
    fs = 30_000.0
    dur = 10.0
    n_chans = 128
    threshold = 2.5
    rate_range = (1, 100)
    chunk_dur = 0.02
    n_times = int(fs * dur)
    frates = np.random.uniform(rate_range[0], rate_range[1], n_chans)
    chunk_len = int(fs * chunk_dur)

    # Create a list of spike times for each channel
    rng = np.random.default_rng()
    spike_offsets = []
    for ch_ix, fr in enumerate(frates):
        lam, size = fs / fr, int(fr * dur)
        isi = rng.poisson(lam=lam, size=size)
        if ch_ix == 0:
            # Add a double spike that should catch refractory period if enabled
            if isi[0] > (chunk_len - 34):
                isi = np.hstack(([chunk_len - 34, 3, 30], isi))
            else:
                isi = np.insert(isi, 1, [3, 30])
        spike_samp_inds = np.cumsum(isi)
        spike_offsets.append(spike_samp_inds[spike_samp_inds < n_times])

    # Create simulated spiking data: white noise + spikes
    in_dat = rng.normal(size=(n_times, n_chans), loc=0, scale=0.1)
    in_dat = np.clip(in_dat, -threshold, threshold)
    for ch_ix, ch_spk_offs in enumerate(spike_offsets):
        in_dat[ch_spk_offs, ch_ix] = threshold + np.random.random(size=(len(ch_spk_offs),))

    bkup_dat = in_dat.copy()
    msg_gen = array_chunker(data=in_dat, chunk_len=chunk_len, axis=0, fs=fs, tzero=0.0)

    # Extract spikes
    transform = threshold_crossing(threshold=threshold, refrac_dur=0.001, return_peak_val=True)
    msgs_out = [transform.send(_) for _ in msg_gen]
    msg_out = AxisArray.concatenate(msgs_out, dim="time")

    expected = np.logical_and(bkup_dat[:-1] < threshold, bkup_dat[1:] >= threshold)
    assert np.allclose(msg_out.data, expected)

    # assert_messages_equal([msg_in], backup)
