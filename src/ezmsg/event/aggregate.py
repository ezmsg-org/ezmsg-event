import typing

import ezmsg.core as ez
from array_api_compat import get_namespace

from ezmsg.sigproc.base import BaseTransformer
from ezmsg.util.messages.axisarray import AxisArray, replace


class AggregateSettings(ez.Settings):
    axis: str
    operator: typing.Literal["sum", "mean"] = "sum"


class Aggregate(BaseTransformer[AggregateSettings, AxisArray, AxisArray]):
    # TODO: Move this to ezmsg-sigproc.aggregate module
    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)
        axis_idx = message.get_axis_idx(self.settings.axis)

        agg_op = getattr(xp, self.settings.operator)
        agg_data = agg_op(message.data, axis=axis_idx)

        new_dims = list(message.dims)
        new_dims.pop(axis_idx)

        new_axes = message.axes.copy()
        if self.settings.axis in new_axes:
            del new_axes[self.settings.axis]

        return replace(
            message,
            data=agg_data,
            dims=new_dims,
            axes=new_axes,
        )