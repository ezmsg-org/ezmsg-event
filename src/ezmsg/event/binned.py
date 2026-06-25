"""Bin an event stream into a lower-rate count (or rate) signal.

The binning itself is delegated to
:obj:`ezmsg.sigproc.binned_aggregate.BinnedAggregate` so the spike-rate branch
shares a *single* bin-boundary implementation with the spike-band-power branch
(``Pow -> BinnedAggregate(MEAN)``). With ``fractional=True`` (the default) bins
span a fractional ``bin_duration * fs`` samples with a carry accumulator, track
wall-clock time, and are labelled with the nominal ``bin_duration`` gain --
identical to how :obj:`ezmsg.event.rate.EventRate` is consumed downstream. That
makes the two branches land on the same grid at any input rate (including the
off-nominal ~30012 Hz of real recordings), so a downstream ``Merge`` aligns them
with no post-hoc reconciler.

Sparse ``sparse.COO`` inputs (the default ``ThresholdCrossing`` output) are
densified to per-sample contributions before binning; dense inputs are used as
is. Set ``scale_by_value=True`` to weight each event by its stored value instead
of counting occurrences, and ``scale_output=True`` to divide the per-bin count
by ``bin_duration`` (events/second).
"""

import ezmsg.core as ez
import sparse
from array_api_compat import get_namespace
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.sigproc.aggregate import AggregationFunction
from ezmsg.sigproc.binned_aggregate import BinnedAggregateSettings, BinnedAggregateTransformer
from ezmsg.util.messages.axisarray import AxisArray, replace


class BinnedEventAggregatorSettings(ez.Settings):
    bin_duration: float = 0.05
    """Duration of each output bin in seconds."""

    scale_output: bool = True
    """If True, divide each bin's count by ``bin_duration`` (events/second)."""

    axis: str = "time"
    """Name of the axis to bin along."""

    fractional: bool = True
    """If True (default), share ``EventRate``'s fractional wall-clock grid via
    :obj:`BinnedAggregate` (nominal ``bin_duration`` gain). If False, use a fixed
    ``round(bin_duration * fs)`` sample-locked grid. See :obj:`BinnedAggregate`."""

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
            binned = replace(binned, data=binned.data / self.settings.bin_duration)
        return binned


class BinnedEventAggregatorUnit(
    BaseTransformerUnit[BinnedEventAggregatorSettings, AxisArray, AxisArray, BinnedEventAggregator]
):
    SETTINGS = BinnedEventAggregatorSettings


def binned_event_aggregator(
    bin_duration: float = 0.05,
    scale_output: bool = True,
    axis: str = "time",
    fractional: bool = True,
    scale_by_value: bool = False,
) -> BinnedEventAggregator:
    return BinnedEventAggregator(
        BinnedEventAggregatorSettings(
            bin_duration=bin_duration,
            scale_output=scale_output,
            axis=axis,
            fractional=fractional,
            scale_by_value=scale_by_value,
        )
    )
