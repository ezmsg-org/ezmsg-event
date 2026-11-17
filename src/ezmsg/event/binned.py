"""Bin an event stream into a lower-rate count (or rate) signal.

The binning is delegated to
:obj:`ezmsg.sigproc.binned_aggregate.BinnedAggregate` so this shares one
bin-boundary implementation with every other consumer of that binner. With
``fractional=True`` (the default) bins span a fractional ``bin_duration * fs``
samples with a carry accumulator and are labelled with the nominal
``bin_duration`` gain; with ``fractional=False`` they span a fixed
``int(bin_duration * fs)`` samples (sample-locked, matching
:obj:`ezmsg.sigproc.window.Window`). Because the grid comes from the shared
binner, two streams binned this way at the same ``bin_duration`` land on the
same grid for any input rate and can be aligned downstream (e.g. with
``ezmsg.sigproc.merge.Merge``).

Sparse ``sparse.COO`` inputs (e.g. the default
:obj:`ezmsg.event.peak.ThresholdCrossing` output) are densified to per-sample
contributions before binning; dense inputs are used as is. Set
``scale_by_value=True`` to weight each event by its stored value instead of
counting occurrences, and ``scale_output=True`` to divide each bin's count by
that bin's actual duration (events/second).
"""

import ezmsg.core as ez
import numpy as np
import sparse
from array_api_compat import get_namespace
from ezmsg.baseproc import (
    BaseTransformer,
    BaseTransformerUnit,
    suppress_axis_deprecation,
    warn_axis_deprecated,
)
from ezmsg.sigproc.aggregate import AggregationFunction
from ezmsg.sigproc.binned_aggregate import BinnedAggregateSettings, BinnedAggregateTransformer
from ezmsg.util.messages.axisarray import AxisArray, replace


class BinnedEventAggregatorSettings(ez.Settings):
    bin_duration: float = 0.05
    """Duration of each output bin in seconds."""

    scale_output: bool = True
    """If True, divide each bin's count by its actual duration, its sample count / fs (events/second).
    That is not exactly ``bin_duration``: fractional bins differ by a sample, and sample-locked bins
    span ``int(bin_duration * fs)`` samples."""

    axis: str | None = None
    """.. deprecated:: 1.4
        Scheduled for removal in 2.0. Binning carries an open partial bin across
        message boundaries, which is only meaningful along the dimension messages
        accumulate along; that dimension now comes from
        :attr:`~ezmsg.util.messages.axisarray.AxisArray.stream_dim`."""

    def __post_init__(self) -> None:
        warn_axis_deprecated(self, package="ezmsg-event", removal="2.0")

    fractional: bool = True
    """If True (default), bins span a fractional ``bin_duration * fs`` samples via
    :obj:`BinnedAggregate` and are labelled with the nominal ``bin_duration``
    gain. If False, bins span a fixed ``int(bin_duration * fs)`` samples
    (sample-locked). See :obj:`BinnedAggregate`."""

    scale_by_value: bool = False
    """If True, weight each event by its stored value; if False (default), every
    nonzero entry contributes a count of 1."""


class BinnedEventAggregator(BaseTransformer[BinnedEventAggregatorSettings, AxisArray, AxisArray]):
    """Count events per fixed-duration bin, delegating binning to sigproc.

    The per-bin reduction, carry across message boundaries, and output time axis
    all come from :obj:`BinnedAggregateTransformer`; this wrapper only converts
    events to per-sample contributions and optionally rate-normalizes.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Forwarding this stage's own already-warned setting; warning again would
        # name a sigproc class for something the user set on ours.
        with suppress_axis_deprecation():
            self._binner = BinnedAggregateTransformer(
                BinnedAggregateSettings(
                    axis=self.settings.axis,
                    bin_duration=self.settings.bin_duration,
                    operation=AggregationFunction.SUM,
                    fractional=self.settings.fractional,
                )
            )

    def _process(self, message: AxisArray) -> AxisArray:
        data = message.data
        if isinstance(data, sparse.SparseArray):
            data = data.todense()
        xp = get_namespace(data)
        # Per-sample contribution: the event value, or 1 per nonzero entry.
        # float64 where usable (exact integer counts; matches the legacy output
        # dtype); float32 otherwise. MLX *exposes* ``mx.float64`` as an attribute
        # but the GPU rejects it ("float64 is not supported on the GPU"), so
        # ``hasattr(xp, "float64")`` is not a sufficient capability check --
        # detect the MLX namespace explicitly and fall back to float32.
        is_mlx = getattr(xp, "__name__", "") == "mlx.core"
        float_dtype = xp.float64 if (hasattr(xp, "float64") and not is_mlx) else xp.float32
        contrib = data if self.settings.scale_by_value else (data != 0)
        contrib = contrib.astype(float_dtype)

        binned = self._binner(replace(message, data=contrib))

        if self.settings.scale_output and binned.data.size:
            # The bins just closed are the schedule's last n; bin m spans samples [int(m*spb), int((m+1)*spb)).
            sched = self._binner.state.schedule
            ax = binned.get_axis_idx(self._binner.state.axis)
            n = binned.data.shape[ax]
            edges = (np.arange(sched.n_bins_done - n, sched.n_bins_done + 1) * sched.spb).astype(np.int64)
            # A bin with no samples (bin_duration under one sample) holds a count of 0; keep it 0, not NaN.
            durations = np.maximum(np.diff(edges), 1) / sched.fs
            shape = [1] * binned.data.ndim
            shape[ax] = n
            binned = replace(binned, data=binned.data / xp.asarray(durations.reshape(shape), dtype=binned.data.dtype))
        return binned

    def flush(self, message: AxisArray | None = None) -> AxisArray | None:
        """Process ``message`` if given, then emit the open partial bin if the output is a rate.

        The binner's own ``flush`` drops a partial SUM bin, so this reads its carry: the count is
        divided by the partial bin's actual duration and labelled with that bin's place on the grid
        and the old stream's segment, config_id and attrs. The next message starts a fresh grid.
        """
        out = None if message is None else self(message)
        st = self._binner.state
        # A partial count bin (scale_output=False) is dropped at a gap: it is not comparable with full bins.
        if not self.settings.scale_output or st.carry is None:
            return out
        template = replace(st.template, attrs=st.last_attrs)
        ax = template.get_axis_idx(st.axis)
        n = st.carry.shape[ax]
        if n == 0:
            return out
        gain_in = template.axes[st.axis].gain
        xp = get_namespace(st.carry)
        rate = xp.sum(st.carry, axis=ax, keepdims=True) / (n * gain_in)
        # No new samples: closes no bin, only reports where the open one sits.
        step = st.schedule.advance(n_new=0, in_offset=st.next_offset, gain_in=gain_in)
        tail = replace(
            template,
            data=rate,
            axes={
                **template.axes,
                st.axis: replace(template.axes[st.axis], gain=step.output_gain, offset=step.output_offset),
            },
            config_id=st.configs.update(st.last_config_id),
        )
        st.carry = None
        self._binner._request_reset()
        if out is None or out.data.shape[ax] == 0:
            return tail
        return replace(out, data=xp.concat((out.data, tail.data), axis=ax))


class BinnedEventAggregatorUnit(
    BaseTransformerUnit[BinnedEventAggregatorSettings, AxisArray, AxisArray, BinnedEventAggregator]
):
    SETTINGS = BinnedEventAggregatorSettings
