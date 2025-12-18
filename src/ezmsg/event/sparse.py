from dataclasses import replace

import ezmsg.core as ez
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray


class DensifySettings(ez.Settings):
    pass


class DensifyTransformer(BaseTransformer[DensifySettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        if hasattr(message.data, "todense"):
            return replace(message, data=message.data.todense())
        else:
            return message


class DensifyUnit(BaseTransformerUnit[DensifySettings, AxisArray, AxisArray, DensifyTransformer]):
    SETTINGS = DensifySettings
