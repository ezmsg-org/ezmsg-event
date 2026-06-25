import ezmsg.core as ez
from ezmsg.baseproc import BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.event.binned import BinnedEventAggregator, BinnedEventAggregatorSettings


class EventRateSettings(ez.Settings):
    bin_duration: float = 0.05

    fractional: bool = True
    """If True (default), bins track wall-clock time on the nominal-``bin_duration``
    grid (fractional samples-per-bin with a carry accumulator). If False, bins are
    a fixed ``round(bin_duration * fs)`` samples (sample-locked). See
    :obj:`ezmsg.sigproc.binned_aggregate.BinnedAggregate`."""


class Rate(BinnedEventAggregator):
    """
    Event rate calculator (events per second).

    Counts events per bin and divides by ``bin_duration`` to get rate in
    events/second. Binning is delegated to
    :obj:`ezmsg.sigproc.binned_aggregate.BinnedAggregate`, so the spike-rate
    output shares one bin-boundary implementation -- and therefore the same
    output grid -- with the spike-band-power branch.
    """

    def __init__(self, settings: EventRateSettings) -> None:
        super().__init__(
            BinnedEventAggregatorSettings(
                bin_duration=settings.bin_duration,
                scale_output=True,  # counts -> events/second
                axis="time",
                fractional=settings.fractional,
                scale_by_value=False,  # count events, ignore stored peak values
            )
        )


class EventRate(BaseTransformerUnit[EventRateSettings, AxisArray, AxisArray, Rate]):
    """Unit for computing event rate from sparse (or dense) events."""

    SETTINGS = EventRateSettings
