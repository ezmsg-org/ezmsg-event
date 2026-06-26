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
counting occurrences, and ``scale_output=True`` to divide the per-bin count by
``bin_duration`` (events/second).
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
