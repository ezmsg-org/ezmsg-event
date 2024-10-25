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
    min_width: int = 1  # Consider making this a parameter.
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
                # We will need a copy of the raw data to know what the values were at the peak
                data_raw = msg_in.data.copy()
            msg_in = scaler.send(msg_in)
        elif return_peak_val:
            data_raw = msg_in.data

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
        # TODO: If we need peak value, then we also need to prepend previous raw values.
        # if align_on_peak or return_peak_val:
        #     data = np.concatenate((_data, data), axis=0)

        # Find threshold crossing where sample k is not over and sample k+1 is over.
        b_cross_over = np.logical_and(overs[:-1] != 1, overs[1:] == 1)
        feat_idx, samp_idx = np.where(b_cross_over.T)
        samp_idx += 1  # Because we looked at samples 1:

        # Get a ragged array of event indices per feature
        uq_feats, feat_splits = np.unique(feat_idx, return_index=True)
        feat_crosses: dict[int, npt.NDArray] = {k: v for k, v in zip(uq_feats, np.split(samp_idx, feat_splits[1:]))}

        # Optionally drop crossings during refractory period
        # TODO: Consider putting this into its own unit. The downside is some of the subsequent computation of peak
        #  loc / value for the to-be-dropped refractory events would otherwise be avoided.
        if refrac_width > 2:
            for f_idx, s_idx in feat_crosses.items():
                # Calculate inter-event intervals
                iei = np.diff(s_idx)
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
                    feat_crosses[f_idx] = s_idx[0] + np.hstack((0, np.cumsum(iei)))
                    # Reset the b_cross_over array for this feature
                    b_cross_over[:, f_idx] = False
                    b_cross_over[feat_crosses[f_idx] - 1, f_idx] = True

            # TODO: Can we produce feat_idx and samp_idx faster from feat_crosses?
            feat_idx, samp_idx = np.where(b_cross_over.T)
            samp_idx += 1

        # Last part of (intermediate or final) outputs
        result_val = np.ones(samp_idx.shape, dtype=bool)

        if not (align_on_peak or return_peak_val):
            # We don't care where the peak is, only that we crossed threshold
            n_pad = 0
            # TODO: Keep the last sample only overs and values for the next iteration.
        else:
            # Calculate peak location and value.
            # Pad using last sample until last cross has at least max_width following samples.
            n_pad = max(0, max(samp_idx) + max_width - overs.shape[0])
            overs = np.pad(overs, ((0, n_pad), (0, 0)), mode="edge")
            # extract `overs` vectors for each event.
            s_idx = np.arange(max_width)[None, :] + samp_idx[:, None]
            ep_overs = overs[s_idx, feat_idx[:, None]]  # (n_events, max_width)

            # We will only consider events that crossed back
            b_ev_crossback = np.any(ep_overs[:, 1:] != 1, axis=1)

            # TODO: First we need to store events that failed to cross back for the next iteration
            # Note: this buffer can accumulate indefinitely if we have a dead signal or one that
            # that is constantly rising, so we drop crossovers that haven't had a cross back
            # after some limit.

            # Now resume processing events that crossed back.
            if np.any(b_cross_over) and np.any(b_ev_crossback):
                # Find the event lengths: i.e., the first non-over-threshold value for each event.
                ev_len = ep_overs[b_ev_crossback, 1:].argmin(axis=1)
                ev_len += 1

                # Only keep events that had the right width
                b_width = np.logical_and(ev_len >= min_width, ev_len <= max_width)
                ev_len = ev_len[b_width]
                samp_idx = samp_idx[b_ev_crossback][b_width]
                feat_idx = feat_idx[b_ev_crossback][b_width]

                # For each unique event-length, extract the peak data and find the largest value and index.
                pk_offset = np.zeros_like(ev_len)
                uq_lens, len_grps = np.unique(ev_len, return_inverse=True)
                for len_idx, ep_len in enumerate(uq_lens):
                    b_grp = len_grps == len_idx
                    ep_resamp = np.arange(ep_len)[None, :] + samp_idx[b_grp, None]
                    eps = data[ep_resamp, feat_idx[b_grp, None]]
                    if threshold >= 0:
                        pk_offset[b_grp] = np.argmax(eps, axis=1)
                    else:
                        pk_offset[b_grp] = np.argmin(eps, axis=1)
                if return_peak_val:
                    result_val = data_raw[samp_idx + pk_offset, feat_idx]
                if align_on_peak:
                    samp_idx += pk_offset

        if return_sparse_mat:
            # Prepare sparse matrix output
            result = scipy.sparse.csr_array((result_val, (samp_idx, feat_idx)))
            msg_out = AxisArray(result, dims=["time", "ch"], axes={"time": msg_in.axes["time"]})
        else:
            # TODO: If input ndim > 2 then we need to reinterpret feat_idx
            # Prepare EventMessage output
            samp_times = msg_in.axes["time"].offset + msg_in.axes["time"].gain * samp_idx
            msg_out = []
            for t, ch, val in zip(samp_times, feat_idx, result_val):
                msg_out.append(EventMessage(t, int(ch), 0, val))
