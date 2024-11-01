from dataclasses import replace
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.sigproc.base import GenAxisArray
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import AxisArray


@consumer
def to_dense() -> typing.Generator[AxisArray, AxisArray, None]:
    msg_out = AxisArray(np.array([]), dims=[""])
    while True:
        msg_in: AxisArray = yield msg_out
        if hasattr(msg_in.data, "todense"):
            msg_out = replace(msg_in, data=msg_in.data.todense())
        else:
            msg_out = msg_in


class DensifySettings(ez.Settings):
    pass


class Densify(GenAxisArray):
    """:obj:`Unit` for :obj:`bandpower`."""

    SETTINGS = DensifySettings

    def construct_generator(self):
        self.STATE.gen = to_dense()
