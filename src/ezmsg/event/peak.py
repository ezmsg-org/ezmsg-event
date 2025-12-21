"""
Detects peaks in a signal.
"""

import typing

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import sparse
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.sigproc.scaler import AdaptiveStandardScalerTransformer
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import AxisArray, replace  # slice_along_axis,

from .message import EventMessage


class ThresholdSettings(ez.Settings):
    threshold: float = -3.5
    """the value the signal must cross before the peak is found."""

    max_peak_dur: float = 0.002
    """The maximum duration of a peak in seconds."""

    min_peak_dur: float = 0.0
    """The minimum duration of a peak in seconds. If 0 (default), no minimum duration is enforced."""

    refrac_dur: float = 0.001
    """The minimum duration between peaks in seconds. If 0 (default), no refractory period is enforced."""

    align_on_peak: bool = False
    """If False (default), the returned sample index indicates the first sample across threshold.
    If True, the sample index indicates the sample with the largest deviation after threshold crossing."""

    return_peak_val: bool = False
    """If True then the peak value is included in the EventMessage or sparse matrix payload."""

    auto_scale_tau: float = 0.0
    """If > 0, the data will be passed through a standard scaler prior to thresholding."""


@processor_state
class ThresholdCrossingState:
    """State for ThresholdCrossingTransformer."""

    max_width: int = 0

    min_width: int = 1

    refrac_width: int = 0

    scaler: AdaptiveStandardScalerTransformer | None = None
    """Object performing adaptive z-scoring."""

    data: npt.NDArray | None = None
    """Trailing buffer in case peak spans sample chunks. Only used if align_on_peak or return_peak_val."""

    data_raw: npt.NDArray | None = None
    """Keep track of the raw data so we can return_peak_val. Only needed if using the scaler."""

    elapsed: npt.NDArray | None = None
    """Track number of samples since last event for each feature. Used especially for refractory period."""


class ThresholdCrossingTransformer(
    BaseStatefulTransformer[ThresholdSettings, AxisArray, AxisArray, ThresholdCrossingState]
):
    """Transformer that detects threshold crossing events."""

    def _hash_message(self, message: AxisArray) -> int:
        ax_idx = message.get_axis_idx("time")
        sample_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1 :]
        return hash((message.key, sample_shape, message.axes["time"].gain))

    def _reset_state(self, message: AxisArray) -> None:
        """Reset the state variables."""
        ax_idx = message.get_axis_idx("time")

        # Precalculate some simple math we'd otherwise have to calculate on every iteration.
        fs = 1 / message.axes["time"].gain
        self._state.max_width = int(self.settings.max_peak_dur * fs)
        self._state.min_width = int(self.settings.min_peak_dur * fs)
        self._state.refrac_width = int(self.settings.refrac_dur * fs)

        # We'll need the first sample (keep time dim!) for a few of our state initializations
        data = np.moveaxis(message.data, ax_idx, -1)
        first_samp = data[..., :1]

        # Prepare optional state variables
        self._state.scaler = None
        self._state.data_raw = None
        if self.settings.auto_scale_tau > 0:
            self._state.scaler = AdaptiveStandardScalerTransformer(
                time_constant=self.settings.auto_scale_tau, axis="time"
            )
            if self.settings.return_peak_val:
                self._state.data_raw = first_samp

        # We always need at least the previous iteration's last sample for tracking whether we are newly over threshold,
        #  and potentially for aligning on peak or returning the peak value.
        self._state.data = first_samp if self._state.scaler is None else np.zeros_like(first_samp)

        # Initialize the count of samples since last event for each feature. We initialize at refrac_width+1
        #  to ensure that even the first sample is eligible for events.
        self._state.elapsed = np.zeros((np.prod(data.shape[:-1]),), dtype=int) + (self._state.refrac_width + 1)

    def _process(self, message: AxisArray) -> AxisArray:
        """
        Process incoming samples and detect threshold crossings.

        Args:
            msg: The input AxisArray containing signal data

        Returns:
            AxisArray with sparse data containing detected events
        """
        ax_idx = message.get_axis_idx("time")

        # If the time axis is not the last axis, we need to move it to the end.
        if ax_idx != (message.data.ndim - 1):
            message = replace(
                message,
                data=np.moveaxis(message.data, ax_idx, -1),
                dims=message.dims[:ax_idx] + message.dims[ax_idx + 1 :] + ["time"],
            )

        # Take a copy of the raw data if needed and prepend to our state data_raw
        #  This will only exist if we are autoscaling AND we need to capture the true peak value.
        if self._state.data_raw is not None:
            self._state.data_raw = np.concatenate((self._state.data_raw, message.data), axis=-1)

        # Run the message through the standard scaler if needed. Note: raw value is lost unless we copied it above.
        if self._state.scaler is not None:
            message = self._state.scaler(message)

        # Prepend the previous iteration's last (maybe z-scored) sample to the current (maybe z-scored) data.
        data = np.concatenate((self._state.data, message.data), axis=-1)
        # Take note of how many samples were prepended. We will need this later when we modify `overs`.
        n_prepended = self._state.data.shape[-1]

        # Identify which data points are over threshold
        overs = data >= self.settings.threshold if self.settings.threshold >= 0 else data <= self.settings.threshold

        # Find threshold _crossing_: where sample k is over and sample k-1 is not over.
        b_cross_over = np.logical_and(overs[..., 1:], ~overs[..., :-1])
        cross_idx = list(np.where(b_cross_over))  # List of indices into each dim
        # We ignored the first sample when looking for crosses so we increment the sample index by 1.
        cross_idx[-1] += 1

        # Note: There is an assumption that the 0th sample only serves as a reference and is not part of the output;
        #  this will be trimmed at the very end. For now the offset is useful for bookkeeping (peak finding, etc.).

        # Optionally drop crossings during refractory period
        # TODO: This should go in its own transformer.
        #  However, a general purpose refractory-period-enforcer would keep track of its own event history,
        #  so we would probably do this step before prepending with historical samples.
        if self._state.refrac_width > 2 and len(cross_idx[-1]) > 0:
            # Find the unique set of features that have at least one cross-over,
            # and the indices of the first crossover for each.
            ravel_feat_inds = np.ravel_multi_index(cross_idx[:-1], overs.shape[:-1])
            uq_feats, feat_splits = np.unique(ravel_feat_inds, return_index=True)
            # Calculate the inter-event intervals (IEIs) for each feature. First get all the IEIs.
            ieis = np.diff(np.hstack(([cross_idx[-1][0] + 1], cross_idx[-1])))
            # Then reset the interval at feature boundaries.
            ieis[feat_splits] = cross_idx[-1][feat_splits] + self._state.elapsed[uq_feats]
            b_drop = ieis <= self._state.refrac_width
            drop_idx = np.where(b_drop)[0]
            final_drop = []
            while len(drop_idx) > 0:
                d_idx = drop_idx[0]
                # Update next iei so its interval refers to the event before the to-be-dropped event.
                #  but only if the next iei belongs to the same feature.
                if ((d_idx + 1) < len(ieis)) and (d_idx + 1) not in feat_splits:
                    ieis[d_idx + 1] += ieis[d_idx]
                # We will later remove this event from samp_idx and cross_idx
                final_drop.append(d_idx)
                # Remove the dropped event from drop_idx.
                drop_idx = drop_idx[1:]

                # If the next event is now outside the refractory period then it will not be dropped.
                if len(drop_idx) > 0 and ieis[drop_idx[0]] > self._state.refrac_width:
                    drop_idx = drop_idx[1:]
            cross_idx = [np.delete(_, final_drop) for _ in cross_idx]

        # Calculate the 'value' at each event.
        hold_idx = overs.shape[-1] - 1
        if len(cross_idx[-1]) == 0:
            # No events; not values to calculate.
            result_val = np.ones(
                cross_idx[-1].shape,
                dtype=data.dtype if self.settings.return_peak_val else bool,
            )
        elif not (self._state.min_width > 1 or self.settings.align_on_peak or self.settings.return_peak_val):
            # No postprocessing required. TODO: Why is min_width <= 1 a requirement here?
            result_val = np.ones(cross_idx[-1].shape, dtype=bool)
        else:
            # Do postprocessing of events: width-checking, align-on-peak, and/or include peak value in return.
            # Each of these requires finding the true peak, which requires pulling out a snippet around the
            #  threshold crossing event.
            # We extract max_width-length vectors of `overs` values for each event. This might require some padding
            #  if the event is near the end of the data. Pad with the last sample until the expected end of the event.
            n_pad = max(0, max(cross_idx[-1]) + self._state.max_width - overs.shape[-1])
            pad_width = ((0, 0),) * (overs.ndim - 1) + ((0, n_pad),)
            overs_padded = np.pad(overs, pad_width, mode="edge")

            # Extract the segments for each event.
            # First we get the sample indices. This is 2-dimensional; first dim for offset and remaining for seg length.
            s_idx = np.arange(self._state.max_width)[None, :] + cross_idx[-1][:, None]
            # Combine feature indices and time indices to extract segments of overs.
            #  Note: We had to expand each of our feature indices also be 2-dimensional
            # -> ndarray (eat dims ..., max_width)
            ep_overs = overs_padded[tuple(_[:, None] for _ in cross_idx[:-1]) + (s_idx,)]

            # Find the event lengths: i.e., the first non-over-threshold value for each event.
            # Warning: Values are invalid for events that don't cross back.
            ev_len = ep_overs[..., 1:].argmin(axis=-1)
            ev_len += 1  # Adjust because we skipped first sample

            # Identify peaks that successfully cross back
            b_ev_crossback = np.any(~ep_overs[..., 1:], axis=-1)

            if self._state.min_width > 1:
                # Drop events that have crossed back but fail min_width
                b_long = ~np.logical_and(b_ev_crossback, ev_len < self._state.min_width)
                cross_idx = tuple(_[b_long] for _ in cross_idx)
                ev_len = ev_len[b_long]
                b_ev_crossback = b_ev_crossback[b_long]

            # We are returning a sparse array and unfinished peaks must be buffered for the next iteration.
            # Find the earliest unfinished event. If none, we still buffer the final sample.
            b_unf = ~b_ev_crossback
            hold_idx = cross_idx[-1][b_unf].min() if np.any(b_unf) else hold_idx

            # Trim events that are past the hold_idx. They will be processed next iteration.
            b_pass_ev = cross_idx[-1] < hold_idx
            cross_idx = [_[b_pass_ev] for _ in cross_idx]
            ev_len = ev_len[b_pass_ev]

            if np.any(b_unf):
                # Must hold back at least 1 sample before start of unfinished events so we can re-detect.
                hold_idx = max(hold_idx - 1, 0)

            # If we are not returning peak values, we can just return bools at the event locations.
            result_val = np.ones(cross_idx[-1].shape, dtype=bool)

            # For remaining _finished_ peaks, get the peak location -- for alignment or if returning its value.
            if self.settings.align_on_peak or self.settings.return_peak_val:
                # We process peaks in batches based on their length, otherwise short peaks could give
                #  incorrect argmax results.
                # TODO: Check performance of using a masked array instead. Might take longer to create the mask.
                pk_offset = np.zeros_like(ev_len)
                uq_lens, len_grps = np.unique(ev_len, return_inverse=True)
                for len_idx, ep_len in enumerate(uq_lens):
                    b_grp = len_grps == len_idx
                    ep_resamp = np.arange(ep_len)[None, :] + cross_idx[-1][b_grp, None]
                    ep_inds_tuple = tuple(_[b_grp, None] for _ in cross_idx[:-1]) + (ep_resamp,)
                    eps = data[ep_inds_tuple]
                    if self.settings.threshold >= 0:
                        pk_offset[b_grp] = np.argmax(eps, axis=1)
                    else:
                        pk_offset[b_grp] = np.argmin(eps, axis=1)

                if self.settings.align_on_peak:
                    # We want to align on the peak, so add the peak offset.
                    cross_idx[-1] += pk_offset

                if self.settings.return_peak_val:
                    # We need the actual peak value.
                    peak_inds_tuple = (
                        tuple(cross_idx)
                        if self.settings.align_on_peak
                        else tuple(cross_idx[:-1]) + (cross_idx[-1] + pk_offset,)
                    )
                    result_val = (self._state.data_raw if self._state.data_raw is not None else data)[peak_inds_tuple]

        # Save data for next iteration
        self._state.data = data[..., hold_idx:]
        if self._state.data_raw is not None:
            # Likely because we are using the scaler, we need a separate copy of the raw data.
            self._state.data_raw = self._state.data_raw[..., hold_idx:]
        # Clear out `elapsed` by adding the max number of samples since the last event.
        self._state.elapsed += hold_idx
        # Yet for features that actually had events, replace the elapsed time with the actual event time
        self._state.elapsed[tuple(cross_idx[:-1])] = hold_idx - cross_idx[-1]
        #  Note: multiple-write to same index ^ is fine because it is sorted and the last value for each is correct.

        # Prepare sparse matrix output
        # Note: The first of the held back samples for next iteration is part of this iteration's return.
        #  Likewise, the first prepended sample on this iteration was part of the previous iteration's return.
        n_out_samps = hold_idx
        t0 = message.axes["time"].offset - (n_prepended - 1) * message.axes["time"].gain
        cross_idx[-1] -= 1  # Discard first prepended sample.
        result = sparse.COO(
            cross_idx,
            data=result_val,
            shape=data.shape[:-1] + (n_out_samps,),
        )
        msg_out = replace(
            message,
            data=result,
            axes={
                **message.axes,
                "time": replace(message.axes["time"], offset=t0),
            },
        )
        return msg_out


class ThresholdCrossing(BaseTransformerUnit[ThresholdSettings, AxisArray, AxisArray, ThresholdCrossingTransformer]):
    SETTINGS = ThresholdSettings


# Legacy API support
@consumer
def threshold_crossing(
    threshold: float = -3.5,
    max_peak_dur: float = 0.002,
    refrac_dur: float = 0.001,
    align_on_peak: bool = False,
    return_peak_val: bool = False,
    auto_scale_tau: float = 0.0,
) -> typing.Generator[list[EventMessage] | AxisArray, AxisArray, None]:
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
    transformer = ThresholdCrossingTransformer(
        threshold=threshold,
        max_peak_dur=max_peak_dur,
        refrac_dur=refrac_dur,
        align_on_peak=align_on_peak,
        return_peak_val=return_peak_val,
        auto_scale_tau=auto_scale_tau,
    )

    msg_out = AxisArray(np.array([]), dims=[""])

    while True:
        msg_in = yield msg_out
        msg_out = transformer(msg_in)
