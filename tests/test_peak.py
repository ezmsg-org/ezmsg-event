import os
import tempfile
from pathlib import Path

import ezmsg.core as ez
import numpy as np
import pytest
import sparse
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger
from ezmsg.util.messages.chunker import ArrayChunker, array_chunker
from ezmsg.util.terminate import TerminateOnTotal

from ezmsg.event.peak import ThresholdCrossing, threshold_crossing
from ezmsg.event.util.simulate import generate_white_noise_with_events


@pytest.mark.parametrize("return_peak_val", [True, False])
def test_threshold_crossing(return_peak_val: bool):
    fs = 30_000.0
    dur = 10.0
    n_chans = 128
    threshold = 2.5
    rate_range = (1, 100)
    chunk_dur = 0.02
    refrac_dur = 0.001
    refrac_width = int(fs * refrac_dur)
    chunk_len = int(fs * chunk_dur)

    in_dat = generate_white_noise_with_events(fs, dur, n_chans, rate_range, chunk_dur, threshold)

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

    final_arr = sparse.concatenate([_.data for _ in msgs_out], axis=1)
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


def get_test_fn(test_name: str | None = None, extension: str = "txt") -> Path:
    """PYTEST compatible temporary test file creator"""
    if test_name is None:
        test_name = os.environ.get("PYTEST_CURRENT_TEST")
        if test_name is not None:
            test_name = test_name.split(":")[-1].split(" ")[0]
        else:
            test_name = __name__

    file_path = Path(tempfile.gettempdir())
    file_path = file_path / Path(f"{test_name}.{extension}")

    # Create the file
    with open(file_path, "w"):
        pass

    return file_path


def test_system():
    fs = 30_000.0
    dur = 2.0
    n_chans = 128
    threshold = 2.5
    rate_range = (1, 100)
    chunk_dur = 0.02
    chunk_len = int(fs * chunk_dur)

    data = generate_white_noise_with_events(fs, dur, n_chans, rate_range, chunk_dur, threshold)
    test_filename = get_test_fn()

    comps = {
        "SOURCE": ArrayChunker(data, chunk_len, fs=fs),
        "THRESH": ThresholdCrossing(threshold=threshold),
        "SINK": MessageLogger(output=test_filename),
        "TERM": TerminateOnTotal(int(fs * dur / chunk_len)),
    }
    conns = (
        (comps["SOURCE"].OUTPUT_SIGNAL, comps["THRESH"].INPUT_SIGNAL),
        (comps["THRESH"].OUTPUT_SIGNAL, comps["SINK"].INPUT_MESSAGE),
        (comps["SINK"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    messages = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)

    for msg_ix, msg in enumerate(messages):
        assert isinstance(msg.data, sparse.SparseArray)
        assert msg.axes["time"].gain == 1 / fs
        assert np.round(msg.axes["time"].offset, 3) == np.round(msg_ix * chunk_dur, 3)
