
import numpy as np
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.sigproc.base import BaseStatefulTransformer, processor_state


class RefractorySettings(ez.Settings):
    dur: float = 0.001
    """The minimum duration between events in seconds. If 0 (default), no refractory period is enforced."""


@processor_state
class Refractory:
    width: int = 0

    elapsed: npt.NDArray | None = None
    """Track number of samples since last event for each feature."""


class RefractoryTransformer(
    BaseStatefulTransformer[RefractorySettings, AxisArray, AxisArray, Refractory]
):
    def _hash_message(self, message: AxisArray) -> int:
        return super()._hash_message(message)

    def _reset_state(self, message: AxisArray) -> None:
        fs = 1 / message.axes["time"].gain
        self._state.width = int(self.settings.dur * fs)
        ax_idx = message.get_axis_idx("time")
        first_samp = slice_along_axis(message.data, slice(None, 1, None), ax_idx)
        self._state.elapsed = np.zeros(first_samp.shape, dtype=int) + (
            self._state.width + 1
        )

    def _process(self, message: AxisArray) -> AxisArray:
        if self._state.width <= 2:
            return message

        # TODO: Get the sparse indices of the message.data

        if len(samp_idx) <= 0:
            return message

        uq_feats, feat_splits = np.unique(cross_idx[0], return_index=True)
        ieis = np.diff(np.hstack(([samp_idx[0] + 1], samp_idx)))
        # Reset elapsed time at feature boundaries.
        ieis[feat_splits] = samp_idx[feat_splits] + self._state.elapsed[uq_feats]
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

        samp_idx = np.delete(samp_idx, final_drop)
        cross_idx = tuple(np.delete(_, final_drop) for _ in cross_idx)
