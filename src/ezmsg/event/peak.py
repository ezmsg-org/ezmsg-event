"""
Detects peaks in a signal.

.. note::
    This module supports the `Array API standard <https://data-apis.org/array-api/>`_,
    enabling use with NumPy, CuPy, PyTorch, and other compatible array libraries.
    Signal data operations are array-API compliant. Event detection and sparse
    output use NumPy regardless of input backend. Output is always sparse.COO.
"""

import math

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import sparse
from array_api_compat import get_namespace, is_numpy_array
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.sigproc.scaler import AdaptiveStandardScalerTransformer
from ezmsg.util.messages.axisarray import AxisArray, replace  # slice_along_axis,


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
        xp = get_namespace(message.data)
        ax_idx = message.get_axis_idx("time")

        # Precalculate some simple math we'd otherwise have to calculate on every iteration.
        fs = 1 / message.axes["time"].gain
        self._state.max_width = int(self.settings.max_peak_dur * fs)
        self._state.min_width = int(self.settings.min_peak_dur * fs)
        self._state.refrac_width = int(self.settings.refrac_dur * fs)

        # We'll need the first sample (keep time dim!) for a few of our state initializations
        perm = (ax_idx,) + tuple(i for i in range(message.data.ndim) if i != ax_idx)
        data = xp.permute_dims(message.data, perm)
        first_samp = data[:1]

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
        self._state.data = first_samp if self._state.scaler is None else xp.zeros_like(first_samp)

        # Initialize the count of samples since last event for each feature. We initialize at refrac_width+1
        #  to ensure that even the first sample is eligible for events.
        self._state.elapsed = np.zeros((math.prod(data.shape[1:]),), dtype=int) + (self._state.refrac_width + 1)

    def _process(self, message: AxisArray) -> AxisArray:
        """
        Process incoming samples and detect threshold crossings.

        Args:
            msg: The input AxisArray containing signal data

        Returns:
            AxisArray with sparse data containing detected events
        """
        xp = get_namespace(message.data)
        ax_idx = message.get_axis_idx("time")

        # If the time axis is not the last axis, we need to move it to the end.
        if ax_idx != 0:
            perm = (ax_idx,) + tuple(i for i in range(message.data.ndim) if i != ax_idx)
            message = replace(
                message,
                data=xp.permute_dims(message.data, perm),
                dims=["time"] + message.dims[:ax_idx] + message.dims[ax_idx + 1 :],
            )

        # Take a copy of the raw data if needed and prepend to our state data_raw
        #  This will only exist if we are autoscaling AND we need to capture the true peak value.
        if self._state.data_raw is not None:
            self._state.data_raw = xp.concat((self._state.data_raw, message.data), axis=0)

        # Run the message through the standard scaler if needed. Note: raw value is lost unless we copied it above.
        if self._state.scaler is not None:
            message = self._state.scaler(message)

        # Prepend the previous iteration's last (maybe z-scored) sample to the current (maybe z-scored) data.
        data = xp.concat((self._state.data, message.data), axis=0)
        # Take note of how many samples were prepended. We will need this later when we modify `overs`.
        n_prepended = self._state.data.shape[0]

        if data.shape[0] == 0:
            # No data at all (empty buffer + empty message). Return empty sparse output.
            result = sparse.COO(
                np.zeros((data.ndim, 0), dtype=np.int64),
                data=np.array([], dtype=data.dtype if self.settings.return_peak_val else bool),
                shape=data.shape,
            )
            return replace(message, data=result)

        if n_prepended == 0:
            # No reference sample from previous iteration (e.g. first message after an empty-data reset).
            # Duplicate the first sample as the reference, matching the convention that _reset_state
            # stores data[:1] so it gets prepended on the next call.
            data = xp.concat((data[:1], data), axis=0)
            n_prepended = 1
            if self._state.data_raw is not None:
                self._state.data_raw = xp.concat((self._state.data_raw[:1], self._state.data_raw), axis=0)

        # Identify which data points are over threshold
        overs = data >= self.settings.threshold if self.settings.threshold >= 0 else data <= self.settings.threshold

        # Find threshold _crossing_: where sample k is over and sample k-1 is not over.
        b_cross_over = overs[1:] & ~overs[:-1]

        # Convert boolean arrays to numpy for event detection (np.where, lexsort, etc.)
        overs_np = np.asarray(overs) if not is_numpy_array(overs) else overs
        b_cross_over_np = np.asarray(b_cross_over) if not is_numpy_array(b_cross_over) else b_cross_over
        cross_idx = list(np.where(b_cross_over_np))  # List of indices into each dim
        # We ignored the first sample when looking for crosses so we increment the sample index by 1.
        cross_idx[0] += 1
        # Sort events by feature first, then by time within each feature.
        # np.where on a time-first array returns events sorted by time; we need them grouped by feature
        # for the refractory period logic and elapsed tracking to work correctly.
        if len(cross_idx[0]) > 0 and len(cross_idx) > 1:
            sort_order = np.lexsort([cross_idx[0]] + cross_idx[1:][::-1])
            cross_idx = [_[sort_order] for _ in cross_idx]

        # Note: There is an assumption that the 0th sample only serves as a reference and is not part of the output;
        #  this will be trimmed at the very end. For now the offset is useful for bookkeeping (peak finding, etc.).

        # Optionally drop crossings during refractory period
        # TODO: This should go in its own transformer. https://github.com/ezmsg-org/ezmsg-event/issues/12
        #  However, a general purpose refractory-period-enforcer would keep track of its own event history,
        #  so we would probably do this step before prepending with historical samples.
        if self._state.refrac_width > 2 and len(cross_idx[0]) > 0:
            # Find the unique set of features that have at least one cross-over,
            # and the indices of the first crossover for each.
            ravel_feat_inds = np.ravel_multi_index(cross_idx[1:], overs_np.shape[1:])
            uq_feats, feat_splits = np.unique(ravel_feat_inds, return_index=True)
            # Calculate the inter-event intervals (IEIs) for each feature. First get all the IEIs.
            ieis = np.diff(np.hstack(([cross_idx[0][0] + 1], cross_idx[0])))
            # Then reset the interval at feature boundaries.
            ieis[feat_splits] = cross_idx[0][feat_splits] + self._state.elapsed[uq_feats]
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
        hold_idx = overs_np.shape[0] - 1
        if len(cross_idx[0]) == 0:
            # No events; not values to calculate.
            result_val = np.ones(
                cross_idx[0].shape,
                dtype=data.dtype if self.settings.return_peak_val else bool,
            )
        elif not (self._state.min_width > 1 or self.settings.align_on_peak or self.settings.return_peak_val):
            # No postprocessing required. TODO: Why is min_width <= 1 a requirement here?
            result_val = np.ones(cross_idx[0].shape, dtype=bool)
        else:
            # Do postprocessing of events: width-checking, align-on-peak, and/or include peak value in return.
            # Each of these requires finding the true peak, which requires pulling out a snippet around the
            #  threshold crossing event.
            # We extract max_width-length vectors of `overs` values for each event. This might require some padding
            #  if the event is near the end of the data. Pad with the last sample until the expected end of the event.
            n_pad = max(0, max(cross_idx[0]) + self._state.max_width - overs_np.shape[0])
            pad_width = ((0, n_pad),) + ((0, 0),) * (overs_np.ndim - 1)
            overs_padded = np.pad(overs_np, pad_width, mode="edge")

            # Extract the segments for each event.
            # First we get the sample indices. This is 2-dimensional; first dim for offset and remaining for seg length.
            s_idx = np.arange(self._state.max_width)[None, :] + cross_idx[0][:, None]
            # Combine feature indices and time indices to extract segments of overs.
            #  Note: We had to expand each of our feature indices also be 2-dimensional
            # -> ndarray (eat dims ..., max_width)
            ep_overs = overs_padded[(s_idx,) + tuple(_[:, None] for _ in cross_idx[1:])]

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
            hold_idx = cross_idx[0][b_unf].min() if np.any(b_unf) else hold_idx

            # Trim events that are past the hold_idx. They will be processed next iteration.
            b_pass_ev = cross_idx[0] < hold_idx
            cross_idx = [_[b_pass_ev] for _ in cross_idx]
            ev_len = ev_len[b_pass_ev]

            if np.any(b_unf):
                # Must hold back at least 1 sample before start of unfinished events so we can re-detect.
                hold_idx = max(hold_idx - 1, 0)

            # If we are not returning peak values, we can just return bools at the event locations.
            result_val = np.ones(cross_idx[0].shape, dtype=bool)

            # For remaining _finished_ peaks, get the peak location -- for alignment or if returning its value.
            if self.settings.align_on_peak or self.settings.return_peak_val:
                # Convert data to numpy for advanced integer indexing into sparse.COO output.
                data_np = np.asarray(data) if not is_numpy_array(data) else data
                raw_source_np = data_np
                if self._state.data_raw is not None:
                    raw_source_np = (
                        np.asarray(self._state.data_raw)
                        if not is_numpy_array(self._state.data_raw)
                        else self._state.data_raw
                    )
                # We process peaks in batches based on their length, otherwise short peaks could give
                #  incorrect argmax results.
                # TODO: Check performance of using a masked array instead. Might take longer to create the mask.
                pk_offset = np.zeros_like(ev_len)
                uq_lens, len_grps = np.unique(ev_len, return_inverse=True)
                for len_idx, ep_len in enumerate(uq_lens):
                    b_grp = len_grps == len_idx
                    ep_resamp = np.arange(ep_len)[None, :] + cross_idx[0][b_grp, None]
                    ep_inds_tuple = (ep_resamp,) + tuple(_[b_grp, None] for _ in cross_idx[1:])
                    eps = data_np[ep_inds_tuple]
                    if self.settings.threshold >= 0:
                        pk_offset[b_grp] = np.argmax(eps, axis=1)
                    else:
                        pk_offset[b_grp] = np.argmin(eps, axis=1)

                if self.settings.align_on_peak:
                    # We want to align on the peak, so add the peak offset.
                    cross_idx[0] += pk_offset

                if self.settings.return_peak_val:
                    # We need the actual peak value.
                    peak_inds_tuple = (
                        tuple(cross_idx)
                        if self.settings.align_on_peak
                        else (cross_idx[0] + pk_offset,) + tuple(cross_idx[1:])
                    )
                    result_val = raw_source_np[peak_inds_tuple]

        # Save data for next iteration
        self._state.data = data[hold_idx:]
        if self._state.data_raw is not None:
            # Likely because we are using the scaler, we need a separate copy of the raw data.
            self._state.data_raw = self._state.data_raw[hold_idx:]
        # Clear out `elapsed` by adding the max number of samples since the last event.
        self._state.elapsed += hold_idx
        # Yet for features that actually had events, replace the elapsed time with the actual event time
        self._state.elapsed[tuple(cross_idx[1:])] = hold_idx - cross_idx[0]
        #  Note: multiple-write to same index ^ is fine because it is sorted and the last value for each is correct.

        # Prepare sparse matrix output
        # Note: The first of the held back samples for next iteration is part of this iteration's return.
        #  Likewise, the first prepended sample on this iteration was part of the previous iteration's return.
        n_out_samps = hold_idx
        t0 = message.axes["time"].offset - (n_prepended - 1) * message.axes["time"].gain
        cross_idx[0] -= 1  # Discard first prepended sample.
        result = sparse.COO(
            cross_idx,
            data=result_val,
            shape=(n_out_samps,) + data.shape[1:],
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
