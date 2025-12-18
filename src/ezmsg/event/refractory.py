import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import sparse
from ezmsg.baseproc import BaseStatefulTransformer, processor_state
from ezmsg.util.messages.axisarray import AxisArray, replace


class RefractorySettings(ez.Settings):
    dur: float = 0.001
    """The minimum duration between events in seconds. If 0 (default), no refractory period is enforced."""


@processor_state
class Refractory:
    width: int = 0

    elapsed: npt.NDArray | None = None
    """Track number of samples since last event for each feature."""


class RefractoryTransformer(BaseStatefulTransformer[RefractorySettings, AxisArray, AxisArray, Refractory]):
    def _hash_message(self, message: AxisArray) -> int:
        return super()._hash_message(message)

    def _reset_state(self, message: AxisArray) -> None:
        fs = 1 / message.axes["time"].gain
        self._state.width = int(self.settings.dur * fs)
        ax_idx = message.get_axis_idx("time")
        # Get the shape of features (all dims except time)
        feat_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1 :]
        n_feats = int(np.prod(feat_shape))
        self._state.elapsed = np.zeros((n_feats,), dtype=int) + (self._state.width + 1)

    def _process(self, message: AxisArray) -> AxisArray:
        if self._state.width <= 2:
            return message

        ax_idx = message.get_axis_idx("time")
        n_samps = message.data.shape[ax_idx]

        # Get the sparse indices of the message.data
        # coords is a tuple of arrays, one per dimension
        coords = message.data.coords
        if coords.shape[1] == 0:
            # No events, update elapsed and return
            self._state.elapsed += n_samps
            return message

        # Separate time indices from feature indices
        samp_idx = coords[ax_idx]
        feat_dims = list(range(message.data.ndim))
        feat_dims.pop(ax_idx)
        feat_coords = tuple(coords[d] for d in feat_dims)

        # Ravel feature indices to 1D for tracking
        feat_shape = tuple(message.data.shape[d] for d in feat_dims)
        if len(feat_coords) > 0:
            ravel_feat_inds = np.ravel_multi_index(feat_coords, feat_shape)
        else:
            ravel_feat_inds = np.zeros(len(samp_idx), dtype=int)

        # Sort by feature then by time to process events in order
        sort_order = np.lexsort((samp_idx, ravel_feat_inds))
        samp_idx = samp_idx[sort_order]
        ravel_feat_inds = ravel_feat_inds[sort_order]
        feat_coords = tuple(fc[sort_order] for fc in feat_coords)

        # Create cross_idx as list with feature coords first, then time
        cross_idx = list(feat_coords) + [samp_idx]

        uq_feats, feat_splits = np.unique(ravel_feat_inds, return_index=True)
        ieis = np.diff(np.hstack(([samp_idx[0] + 1], samp_idx)))
        # Reset elapsed time at feature boundaries.
        ieis[feat_splits] = samp_idx[feat_splits] + self._state.elapsed[uq_feats]
        b_drop = ieis <= self._state.width
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
            if len(drop_idx) > 0 and ieis[drop_idx[0]] > self._state.width:
                drop_idx = drop_idx[1:]

        samp_idx = np.delete(samp_idx, final_drop)
        cross_idx = [np.delete(_, final_drop) for _ in cross_idx]
        ravel_feat_inds = np.delete(ravel_feat_inds, final_drop)

        # Update elapsed state for all features
        self._state.elapsed += n_samps
        # For features that had events, set elapsed to time since last event
        if len(samp_idx) > 0:
            # Get the last event time for each feature that had events
            uq_final_feats, last_idx = np.unique(ravel_feat_inds[::-1], return_index=True)
            last_idx = len(ravel_feat_inds) - 1 - last_idx
            last_samps = samp_idx[last_idx]
            self._state.elapsed[uq_final_feats] = n_samps - last_samps

        # Build output coordinates in original dimension order
        out_coords = [None] * message.data.ndim
        for i, d in enumerate(feat_dims):
            out_coords[d] = cross_idx[i]
        out_coords[ax_idx] = cross_idx[-1]

        # Get the values for kept events
        kept_mask = np.ones(coords.shape[1], dtype=bool)
        kept_mask[sort_order[final_drop]] = False
        result_data = message.data.data[kept_mask]

        result = sparse.COO(
            out_coords,
            data=result_data,
            shape=message.data.shape,
        )

        return replace(message, data=result)
