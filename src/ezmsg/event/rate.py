import typing

import ezmsg.core as ez
from ezmsg.baseproc import BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.event.kernel_activation import (
    ActivationKernelType,
    BinAggregation,
    BinnedKernelActivation,
    BinnedKernelActivationSettings,
)


class EventRateSettings(ez.Settings):
    bin_duration: float = 0.05

    fractional: bool = True
    """If True (default), bins span a fractional ``bin_duration * fs`` samples and
    the output rate is exactly ``1 / bin_duration``. If False, bins are
    sample-locked to ``int(bin_duration * fs)`` samples (output rate
    ``fs / int(bin_duration * fs)``), matching :obj:`ezmsg.sigproc.window.Window`."""

    phase: float | typing.Callable[[float], float] = 0.0
    """Offset of the bin grid from the stream start, in seconds; a callable gets the
    input sample rate at reset. See :attr:`BinnedKernelActivationSettings.phase`."""


class Rate(BinnedKernelActivation):
    """
    Event rate calculator (events per second).

    Counts events per bin and divides by that bin's actual duration to get rate in events/second.
    """

    def __init__(self, settings: EventRateSettings) -> None:
        super().__init__(
            BinnedKernelActivationSettings(
                kernel_type=ActivationKernelType.COUNT,
                tau=1.0,  # Not used for COUNT
                bin_duration=settings.bin_duration,
                aggregation=BinAggregation.SUM,
                scale_by_value=False,
                normalize=False,
                rate_normalize=True,
                fractional=settings.fractional,
                phase=settings.phase,
            )
        )


class EventRate(BaseTransformerUnit[EventRateSettings, AxisArray, AxisArray, Rate]):
    """Unit for computing event rate from sparse events."""

    SETTINGS = EventRateSettings
