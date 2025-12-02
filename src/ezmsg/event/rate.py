import typing

import ezmsg.core as ez
from ezmsg.sigproc.base import (
    BaseTransformer,
    CompositeProcessor,
    BaseTransformerUnit,
)
from ezmsg.sigproc.aggregate import (
    AggregateTransformer,
    AggregateSettings,
    AggregationFunction,
)
from ezmsg.sigproc.window import WindowTransformer, WindowSettings
from ezmsg.util.messages.axisarray import AxisArray, replace


class EventRateSettings(ez.Settings):
    bin_duration: float = 0.05


class DensifyAndScaleSettings(ez.Settings):
    scale: float = 1.0


class DensifyAndScale(BaseTransformer[DensifyAndScaleSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        return replace(message, data=(message.data.todense() * self.settings.scale))


class RenameAxisSettings(ez.Settings):
    old_axis: str
    new_axis: str


class RenameAxis(BaseTransformer[RenameAxisSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        new_dims = list(message.dims)
        new_axes = dict(message.axes)

        if self.settings.old_axis in new_dims:
            idx = new_dims.index(self.settings.old_axis)
            new_dims[idx] = self.settings.new_axis
            if self.settings.old_axis in new_axes:
                new_axes[self.settings.new_axis] = new_axes.pop(self.settings.old_axis)

        return replace(message, dims=new_dims, axes=new_axes)


class Rate(CompositeProcessor[EventRateSettings, AxisArray, AxisArray]):
    @staticmethod
    def _initialize_processors(
        settings: EventRateSettings,
    ) -> dict[str, typing.Any]:
        return {
            "window": WindowTransformer(
                WindowSettings(
                    axis="time",
                    newaxis="win",
                    window_dur=settings.bin_duration,
                    window_shift=settings.bin_duration,
                    zero_pad_until="none",
                )
            ),
            "aggregate": AggregateTransformer(
                AggregateSettings(axis="time", operation=AggregationFunction.SUM)
            ),
            "rename": RenameAxis(RenameAxisSettings(old_axis="win", new_axis="time")),
            "densify_and_scale": DensifyAndScale(
                DensifyAndScaleSettings(scale=1.0 / settings.bin_duration)
            ),
        }


class EventRate(BaseTransformerUnit[EventRateSettings, AxisArray, AxisArray, Rate]):
    SETTINGS = EventRateSettings
