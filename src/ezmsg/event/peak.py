"""
Detects peaks in a signal.
"""

from dataclasses import replace
import typing

import ezmsg.core as ez
from ezmsg.sigproc.base import GenAxisArray
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import consumer
from ezmsg.sigproc.scaler import scaler_np
import numpy as np
import numpy.typing as npt
import sparse

from .message import EventMessage


@consumer
def threshold_crossing(
    threshold: float = -3.5,
    max_peak_dur: float = 0.002,
    refrac_dur: float = 0.001,
    align_on_peak: bool = False,
    return_peak_val: bool = False,
    auto_scale_tau: float = 0.0,
) -> typing.Generator[
    list[EventMessage] | AxisArray, AxisArray, None
]:
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

    Note: If either align_on_peak or return_peak_val are True then it is necessary to find the actual peak and not
        just the threshold crossing. This will drastically increase the computational demand. It is recommended to
        tune max_peak_dur to a minimal-yet-reasonable value to limit the search space.

    Returns:
        A primed generator object that yields a list of :obj:`EventMessage` objects for every
        :obj:`AxisArray` it receives via `send`.
    """

    msg_out = AxisArray(np.array([]), dims=[""])

    # Initialize state variables
    sample_shape: tuple[int, ...] | None = None
    fs: float | None = None
    max_width: int = 0
    min_width: int = 1  # Consider making this a parameter.
    refrac_width: int = 0

    scaler: typing.Generator[AxisArray, AxisArray, None] | None = None
    # adaptive z-scoring.
    # TODO: This sample-by-sample adaptation is probably overkill. ezmsg-sigproc should add chunk-wise scaler updating.

    _overs: npt.NDArray | None = None  # (n_feats, <=max_width) int == -1 or +1
    # Trailing buffer to track whether the previous sample(s) were past threshold.

    _data: npt.NDArray | None = None  # (n_feats, <=max_width) in_dtype
    # Trailing buffer in case peak spans sample chunks. Only used if align_on_peak or return_peak_val.

    _data_raw: npt.NDArray | None = None  # (n_feats, <=max_width) in_dtype
    # Only used if return_peak_val and scaler is not None

    _elapsed: npt.NDArray | None = None  # (n_feats,) int
    # Number of samples since last event. Used to enforce refractory period across iterations.
    #
    # _n_skip: npt.NDArray | None = None  # (n_feats,) int

    while True:
        msg_in: AxisArray = yield msg_out

        # Extract basic metadata from message
        ax_idx = msg_in.get_axis_idx("time")
        in_sample_shape = msg_in.data.shape[:ax_idx] + msg_in.data.shape[ax_idx + 1 :]
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
            _overs = None
            _data = None
            _data_raw = None
            n_flat_feats = np.prod(sample_shape)
            _elapsed = np.zeros((n_flat_feats,), dtype=int) + (refrac_width + 1)
            # _n_skip = np.zeros((n_flat_feats,), dtype=int)
            # TODO: Support > 2 dim output with pydata.sparse
            other_dim = "*".join([_ for _ in msg_in.dims if _ != "time"])
            out_axes = (
                msg_in.axes.copy()
                if msg_in.data.ndim == 2
                else {"time": msg_in.axes["time"]}
            )
            template = AxisArray(
                sparse.SparseArray((0, 0)), dims=[other_dim, "time"], axes=out_axes
            )

        # Optionally scale data
        data_raw: npt.NDArray | None = None
        if scaler is not None:
            if return_peak_val:
                data_raw = msg_in.data.copy()
            msg_in = scaler.send(msg_in)

        data = msg_in.data

        # Put the time axis in the 0th dim.
        if ax_idx != data.ndim - 1:
            data = np.moveaxis(data, ax_idx, -1)
            if data_raw is not None:
                data_raw = np.moveaxis(data_raw, ax_idx, -1)

        # Flatten the feature dimensions
        if data.ndim > 2:
            data = data.reshape((-1, data.shape[-1]))
            if data_raw is not None:
                data_raw = data_raw.reshape((-1, data_raw.shape[-1]))

        # Check each sample relative to threshold.
        overs = data >= threshold if threshold >= 0 else data <= threshold

        # Prepend our variables with previous iteration's values. We always expect at least 1 sample to carry over.
        overs = np.concatenate(
            (_overs if _overs is not None else overs[..., :1], overs), axis=-1
        )
        # If we need to identify _where_ the peak was then we must prepend previous data values.
        if align_on_peak or return_peak_val:
            data = np.concatenate(
                (_data if _data is not None else data[..., :1], data), axis=-1
            )
        # If we're doing z-scoring but we need the actual peak value then we must prepend previous RAW values too.
        if return_peak_val and scaler is not None:
            data_raw = np.concatenate(
                (_data_raw if _data_raw is not None else data_raw[..., :1], data_raw),
                axis=-1,
            )
        # We will modify _overs later, so for now we take note of how many samples were prepended.
        n_prepended = 1 if _overs is None else _overs.shape[-1]

        # Find threshold crossing where sample k is over and sample k-1 is not over.
        b_cross_over = np.logical_and(~overs[..., :-1], overs[..., 1:])

        feat_idx, samp_idx = np.where(b_cross_over)
        # Because we looked at samples [1:]...
        samp_idx += 1
        # Note: There is an assumption that the 0th sample only serves as a reference and is not part of the output;
        #  this will be trimmed at the very end. For now the offset is useful for bookkeeping (peak finding, etc.).

        # Optionally drop crossings during refractory period
        if refrac_width > 2 and len(samp_idx) > 0:
            # TODO: Consider putting this into its own unit. The downside to moving it out is that some of the remaining
            #  computation in this function is unnecessary for to-be-dropped events.
            uq_feats, feat_splits = np.unique(feat_idx, return_index=True)
            ieis = np.diff(np.hstack(([samp_idx[0] + 1], samp_idx)))
            # Reset elapsed time at feature boundaries.
            ieis[feat_splits] = samp_idx[feat_splits] + _elapsed[uq_feats]
            b_drop = ieis <= refrac_width
            drop_idx = np.where(b_drop)[0]
            final_drop = []
            while len(drop_idx) > 0:
                d_idx = drop_idx[0]
                # Update next iei so its interval refers to the event before the to-be-dropped event.
                #  but only if the next iei belongs to the same feature.
                if ((d_idx + 1) < len(ieis)) and (d_idx + 1) not in feat_splits:
                    ieis[d_idx + 1] += ieis[d_idx]
                # We will later remove this event from samp_idx and feat_idx
                final_drop.append(d_idx)
                # Remove the dropped event from drop_idx.
                drop_idx = drop_idx[1:]

                # If the next event is now outside the refractory period then it will not be dropped.
                # TODO: Maybe we don't want this feature? e.g., if the 3rd in a triplet is not within the refractory
                #  period of the first, but it is within the refractory period of the 2nd which is now dropped,
                #  maybe the 3rd event should still be dropped? If so then we can refactor above to drop all at once.
                if len(drop_idx) > 0 and ieis[drop_idx[0]] > refrac_width:
                    drop_idx = drop_idx[1:]

            samp_idx = np.delete(samp_idx, final_drop)
            feat_idx = np.delete(feat_idx, final_drop)

        hold_idx = overs.shape[-1] - 1
        if len(samp_idx) == 0:
            # No events.
            result_val = np.ones(
                samp_idx.shape, dtype=data.dtype if return_peak_val else bool
            )
        elif not (min_width > 1 or align_on_peak or return_peak_val):
            # No postprocessing required.
            result_val = np.ones(samp_idx.shape, dtype=bool)
        else:
            # Do postprocessing of events.

            # We extract max_width-length vectors of `overs` values for each event.
            # Pad using last sample until last crossover has at least max_width following samples.
            n_pad = max(0, max(samp_idx) + max_width - overs.shape[-1])
            pad_width = ((0, 0),) * (overs.ndim - 1) + ((0, n_pad),)
            overs_padded = np.pad(overs, pad_width, mode="edge")

            # Multi-index to extract the vectors for each event
            s_idx = np.arange(max_width)[None, :] + samp_idx[:, None]
            ep_overs = overs_padded[feat_idx[:, None], s_idx]  # (n_events, max_width)

            # Find the event lengths: i.e., the first non-over-threshold value for each event.
            # Warning: Values are invalid for events that don't cross back.
            ev_len = ep_overs[..., 1:].argmin(axis=-1)
            ev_len += 1

            # Identify peaks that successfully cross back
            b_ev_crossback = np.any(~ep_overs[..., 1:], axis=-1)

            if min_width > 1:
                # Drop events that have crossed back but fail min_width
                b_short = np.logical_and(b_ev_crossback, ev_len < min_width)
                b_long = ~b_short
                samp_idx = samp_idx[b_long]
                feat_idx = feat_idx[b_long]
                ev_len = ev_len[b_long]
                b_ev_crossback = b_ev_crossback[b_long]

            # We are returning a sparse array and unfinished peaks must be buffered for the next iteration.
            # Find the earliest unfinished event. If none, we still buffer the final sample.
            b_unf = ~b_ev_crossback
            hold_idx = samp_idx[b_unf].min() if np.any(b_unf) else hold_idx

            # Trim events that are past the hold_idx. They will be processed next iteration.
            b_pass_ev = samp_idx < hold_idx
            samp_idx = samp_idx[b_pass_ev]
            feat_idx = feat_idx[b_pass_ev]
            ev_len = ev_len[b_pass_ev]

            if np.any(b_unf):
                # Must hold back at least 1 sample before start of unfinished events so we can re-detect.
                hold_idx = max(hold_idx - 1, 0)

            if not return_peak_val:
                result_val = np.ones(samp_idx.shape, dtype=bool)

            # For remaining _finished_ peaks, get the peak location -- for alignment or if returning its value.
            if align_on_peak or return_peak_val:
                # We process peaks in batches based on their length, otherwise short peaks could give
                #  incorrect argmax results.
                # TODO: Check performance of using a masked array instead. Might take longer to create the mask.
                pk_offset = np.zeros_like(ev_len)
                uq_lens, len_grps = np.unique(ev_len, return_inverse=True)
                for len_idx, ep_len in enumerate(uq_lens):
                    b_grp = len_grps == len_idx
                    ep_resamp = np.arange(ep_len)[None, :] + samp_idx[b_grp, None]
                    eps = data[feat_idx[b_grp, None], ep_resamp]
                    if threshold >= 0:
                        pk_offset[b_grp] = np.argmax(eps, axis=1)
                    else:
                        pk_offset[b_grp] = np.argmin(eps, axis=1)

                # Now that we have the offset to the peak for each event, we can find the peak value and/or align.
                if return_peak_val:
                    if scaler is None:
                        result_val = data[feat_idx, samp_idx + pk_offset]
                    else:
                        result_val = data_raw[feat_idx, samp_idx + pk_offset]
                if align_on_peak:
                    samp_idx += pk_offset

        # Save data for next iteration
        _overs = overs[..., hold_idx:]
        if align_on_peak or return_peak_val:
            _data = data[..., hold_idx:]
            if return_peak_val and scaler is not None:
                _data_raw = data_raw[..., hold_idx:]
        _elapsed += hold_idx
        _elapsed[feat_idx] = hold_idx - samp_idx
        # ^ multiple-write to same index is fine because last value is desired.

        # Prepare sparse matrix output
        # Note: The first of the "held" samples is part of this iteration's return.
        #  Likewise, the first prepended sample was part of the previous iteration's return.
        n_out_samps = hold_idx
        t0 = msg_in.axes["time"].offset - (n_prepended - 1) * msg_in.axes["time"].gain
        samp_idx -= 1  # Discard first prepended sample.
        result = sparse.COO(
            (feat_idx, samp_idx),
            data=result_val,
            shape=data.shape[:-1] + (n_out_samps,),
        )
        msg_out = replace(
            template,
            data=result,
            axes={**template.axes, "time": replace(template.axes["time"], offset=t0)},
        )


class ThresholdSettings(ez.Settings):
    threshold: float = -3.5
    max_peak_dur: float = 0.002
    refrac_dur: float = 0.001
    align_on_peak: bool = False
    return_peak_val: bool = False
    auto_scale_tau: float = 0.0


class ThresholdCrossing(GenAxisArray):
    SETTINGS = ThresholdSettings

    def construct_generator(self):
        self.STATE.gen = threshold_crossing(
            threshold=self.SETTINGS.threshold,
            max_peak_dur=self.SETTINGS.max_peak_dur,
            refrac_dur=self.SETTINGS.refrac_dur,
            align_on_peak=self.SETTINGS.align_on_peak,
            return_peak_val=self.SETTINGS.return_peak_val,
            auto_scale_tau=self.SETTINGS.auto_scale_tau,
        )
