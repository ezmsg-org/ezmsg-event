from dataclasses import dataclass
from numbers import Number


@dataclass
class EventMessage:
    offset: float
    """The temporal offset at which the event occurred. This is a float in seconds. The reference point is
    unspecified and depends on the clock the application uses. Most applications will use time.time."""

    ch_idx: int

    sub_idx: int = 0
    """The sub-index of the channel. For Blackrock multi-unit data: 0=unsorted, 1-5 sorted unit, >5=noise"""

    value: Number = 1
    """The value of the event. This can be any number, but is usually an integer, and is often 1 for spikes."""
