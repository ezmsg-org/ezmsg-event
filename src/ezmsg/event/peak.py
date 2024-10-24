"""
Detects peaks in a signal.
"""
import typing

from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import consumer
from ezmsg.sigproc.scaler import scaler_np
import numpy as np
import numpy.typing as npt
import scipy.sparse

from .message import EventMessage


@consumer
def threshold_crossing(
    threshold: float = -3.5,
    max_peak_dur: float = 0.002,
    refrac_dur: float = 0.001,
    align_on_peak: bool = False,
    return_peak_val: bool = False,
    auto_scale_tau: float = 0.0,
    return_sparse_mat: bool = False,
) -> typing.Generator[typing.Union[typing.List[EventMessage], AxisArray], AxisArray, None]:
    """
    Detect threshold crossing events.

    Args:
        threshold: the value the signal must cross before the peak is found.
        max_peak_dur: The maximum duration of a peak in seconds.
        refrac_dur: The minimum duration between peaks in seconds. If 0 (default), no refractory period is enforced.
        align_on_peak: If False (default), the returned sample index indicates the first sample across threshold.
              If True, the sample index indicates the sample with the largest deviation after threshold crossing.
        return_peak_val: If True then the peak value is included in the EventMessage or sparse matrix payload.
        auto_scale_tau: If > 0, the data will be passed through a standard scaler prior to thresholding.
        return_sparse_mat: If False (default), the return value is a list of EventMessage. If True, the return value
          is an AxisArray message with a sparse matrix in its data payload.

    Note: If either align_on_peak or return_peak_val are True then it is necessary to find the actual peak and not
        just the threshold crossing. This will drastically increase the computational demand. It is recommended to
        tune max_peak_dur to a minimal-yet-reasonable value to limit the search space.

    Returns:
        A primed generator object that yields a list of :obj:`EventMessage` objects for every
        :obj:`AxisArray` it receives via `send`.
    """

    msg_out = AxisArray(np.array([]), dims=[""])

    # Initialize state variables
    sample_shape: typing.Optional[typing.Tuple[int, ...]] = None
    fs: typing.Optional[float] = None
    max_width: int = 0
    refrac_width: int = 0

    scaler: typing.Optional[typing.Generator[AxisArray, AxisArray, None]] = None
    # adaptive z-scoring. TODO: This sample-by-sample adaptation is probably overkill. We should
    #  implement a new scaler for chunk-wise updating.

    _overs: typing.Optional[npt.NDArray] = None  # (max_width, n_feats) int == -1 or +1
    # Trailing buffer to track whether the previous sample(s) were past threshold.

    _elapsed: typing.Optional[npt.NDArray] = None  # (n_feats,) int
    # Number of samples since last event. Used to enforce refractory period across iterations.

    _n_skip: typing.Optional[npt.NDArray] = None  # (n_feats,) int

    _data: typing.Optional[npt.NDArray] = None  # (max_width, n_feats) in_dtype
    # Trailing buffer in case peak spans sample chunks. Only used if return_peak_val is True.

    while True:
        msg_in: AxisArray = yield msg_out

        # Extract basic metadata from message
        ax_idx = msg_in.get_axis_idx("time")
        in_sample_shape = msg_in.data.shape[:ax_idx] + msg_in.data.shape[ax_idx + 1:]
        in_fs = 1 / msg_in.axes["time"].gain

        # If metadata has changed substantially, then reset state variables
        b_reset = sample_shape is None or sample_shape != in_sample_shape
        b_reset = b_reset or fs != in_fs
        if b_reset:
            sample_shape = in_sample_shape
            fs = in_fs
            max_width = int(max_peak_dur * fs)
            refrac_width = int(refrac_dur * fs)
            if auto_scale_tau > 0:
                scaler = scaler_np(time_constant=auto_scale_tau, axis="time")
            n_flat_feats = np.prod(sample_shape)
            _overs = np.zeros((0, n_flat_feats), dtype=int)
            _n_skip = np.zeros((n_flat_feats,), dtype=int)
            if align_on_peak or return_peak_val:
                _data = np.zeros((0, n_flat_feats), dtype=msg_in.data.dtype)

        # Optionally scale data
        data_raw: typing.Optional[npt.NDArray] = None
        if scaler is not None:
            if return_peak_val:
                data_raw = msg_in.data.copy()
            msg_in = scaler.send(msg_in)

        data = msg_in.data

        # Put the time axis in the 0th dim.
        if ax_idx != 0:
            data = np.moveaxis(data, ax_idx, 0)
            if data_raw is not None:
                data_raw = np.moveaxis(data_raw, ax_idx, 0)

        # Flatten the feature dimensions
        if data.ndim > 2:
            data = data.reshape((data.shape[0], -1))
            if data_raw is not None:
                data_raw = data_raw.reshape((data_raw.shape[0], -1))

        # Check if each sample is beyond threshold.
        k = -1 if threshold < 0 else 1
        overs = k * (data >= threshold) - k * (data < threshold)

        # Prepend with previous iteration's overs. Must contain at least 1 sample for cross on new sample 0 to work.
        overs = np.concatenate((_overs, overs), axis=0)
        # if align_on_peak or return_peak_val:
        #     data = np.concatenate((_data, data), axis=0)

        # Find threshold crossing where sample k is not over and sample k+1 is over.
        b_cross_over = np.logical_and(overs[:-1] != 1, overs[1:] == 1)
        over_feat_idx, over_idx = np.where(b_cross_over.T)
        over_idx += 1  # Because we looked at samples 1:
        uq_feats, feat_splits = np.unique(over_feat_idx, return_index=True)
        # Get a ragged array of event indices per feature
        feat_crosses: dict[int, npt.NDArray] = {k: v for k, v in zip(uq_feats, np.split(over_idx, feat_splits[1:]))}

        # Optionally drop crossings during refractory period
        # TODO: Consider putting this into its own unit. The downside is some of the subsequent computation of peak
        #  loc / value for the to-be-dropped refractory events would otherwise be avoided.
        if refrac_width > 2:
            for feat_idx, feat_over_idx in feat_crosses.items():
                # Calculate inter-event intervals
                iei = np.diff(feat_over_idx)
                # Find events that are too close to the previous event
                drop_idx = np.where(iei <= refrac_width)[0]
                if len(drop_idx) > 0:
                    while len(drop_idx) > 0:
                        tmp_idx = drop_idx[0]
                        # Update next iei so it refers to the to-be-dropped event.
                        iei[tmp_idx + 1] += iei[tmp_idx]
                        # Remove the dropped event from iei
                        iei = np.delete(iei, tmp_idx)
                        # Remove the dropped event from drop_idx.
                        drop_idx = drop_idx[1:]
                        # See if we can now skip the next event because it is now outside the refractory period.
                        if iei[tmp_idx] > refrac_width:
                            drop_idx = drop_idx[1:]

                    # Reconstruct the ragged array
                    feat_crosses[feat_idx] = feat_over_idx[0] + np.hstack((0, np.cumsum(iei)))
                    # Update the b_cross_over array
                    b_cross_over[:, feat_idx] = False
                    b_cross_over[feat_crosses[feat_idx] - 1, feat_idx] = True

        if not (align_on_peak or return_peak_val):
            # We don't care where the peak is, only that we crossed threshold
            n_pad = 0
            # TODO:
            result_feat_idx, result_samp_idx = np.where(b_cross_over.T)
            result_samp_idx += 1
            result_val = np.ones(result_samp_idx.shape, dtype=bool)
        else:
            # Prepare index arrays for eventual sparse matrix.
            result_samp_idx = []
            result_feat_idx = []
            result_val = []

            over_feat_idx, over_idx = np.where(b_cross_over.T)
            over_idx += 1

            # Calculate peak location and value.
            # Pad using last sample until last cross has at least max_width following samples.
            n_pad = max(0, max(over_idx) + max_width - overs.shape[0])
            overs = np.pad(overs, ((0, n_pad), (0, 0)), mode="edge")
            # extract signal from each cross over to max_width
            t_sample = np.arange(max_width)[None, :] + over_idx[:, None]
            ep_overs = overs[t_sample, over_feat_idx[:, None]]

            # TODO: if EP crosses back then keep it and add to result.
            b_cross_back = np.logical_and(ep_overs[:-1] == 1, ep_overs[1:] != 1)
            t_idx, ev_idx = np.where(b_cross_back)
            t_idx += 1

            # TODO: if EP does not cross back then keep it for next iteration. Remember to NOT keep padding.

            # If we have crossovers and cross backs...
            if np.any(b_cross_over) and np.any(b_cross_back):
                # Find the indices for the over and backs

                back_idx, back_ch_idx = np.where(b_cross_back)
                back_idx += 1
                # Align over-and-back, channel-by-channel
                for ch_idx in np.unique(over_ch_idx):
                    _over = over_idx[over_ch_idx == ch_idx]
                    _back = back_idx[back_ch_idx == ch_idx]
                    _match = np.searchsorted(_over, _back, side="right") - 1
                    _match_over, _match_bk = np.unique(_match, return_index=True)
                    _over, _back = _over[_match_over], _back[_match_bk]
                    _pk_widths = _back - _over
                    _pk_entry_vals = data[_over, ch_idx]
                    _pk_exit_vals = data[_back, ch_idx]
                    b_keep = np.logical_and(
                        _pk_widths > 2,
                        _pk_widths <= max_width
                    )
                    # b_keep = np.logical_and(
                    #     b_keep,
                    #     _pk_exit_vals <= _pk_entry_vals
                    # )
                    _over, _back = _over[b_keep], _back[b_keep]
                    for start, stop in zip(_over, _back):
                        if threshold >= 0:
                            pk_offset = np.argmax(data[start:stop, ch_idx])
                        else:
                            pk_offset = np.argmin(data[start:stop, ch_idx])
                        pk_val = data[start + pk_offset, ch_idx]
                        if (threshold >= 0 and pk_val <= data[start, ch_idx]) or (pk_val >= data[start, ch_idx]):
                            # Peak is smaller than peak-entry. Not a real peak.
                            continue
                        result_idx.append(start + pk_offset)
                        result_ch.append(ch_idx)
                        result_val.append(pk_val)

        # Create output
        result_idx = np.array(result_idx)
        result_ch = np.array(result_ch)
        result_val = np.array(result_val)

        if return_sparse_mat:
            # Prepare sparse matrix output
            result = scipy.sparse.csr_array((result_val, (result_idx, result_ch)))
            msg_out = AxisArray(result, dims=["time", "ch"], axes={"time": msg_in.axes["time"]})
        else:
            # Prepare EventMessage output
            msg_out = []
            for idx, ch, val in zip(result_idx, result_ch, result_val):
                msg_out.append(EventMessage(idx, ch, val))

        # Prep next iteration
        # We want to retain the data from the last cross_over for each channel that does not have a
        #  matching cross back. However, this buffer can accumulate indefinitely if we have a dead signal or one that
        #  rises
        #  that is constantly rising, so we eliminate crossovers that haven't had a cross back
        #  after some limit.
        n_in = data.shape[0]
        offset = (n_in - 1) * np.ones((data.shape[1]), dtype=int)
        if np.any(b_cross_over):
            over_idx, over_ch_idx = np.where(b_cross_over)
            over_idx += 1
            for ch_idx in np.unique(over_ch_idx):
                b_ch = over_ch_idx == ch_idx
                if np.any(b_ch):
                    last_over = over_idx[b_ch][-1]
                    if last_over > result_idx[result_ch == ch_idx][-1]:
                        offset[ch_idx] = last_over

        offset[(n_in - offset) > max_width] = n_in - 1
        _n_skip = n_in - offset - 1
        offset = min(offset)
        _data = data[offset:]
        _overs = overs[offset:]
