from dataclasses import dataclass
from numbers import Number


@dataclass
class EventMessage:
    timestamp: float
    """The time at which the event occurred. This is a float in seconds.
    The clock should usually use time.time() unless otherwise specified."""

    ch_idx: int

    sub_idx: int = 0
    """The sub-index of the channel. This is used for multi-unit data, and is usually 0 for single-unit data."""

    value: Number = 1
    """The value of the event. This can be any number, but is usually an integer, and is often 1 for spikes."""
