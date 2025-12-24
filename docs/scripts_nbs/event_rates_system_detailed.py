import ezmsg.core as ez
import typer
from ezmsg.sigproc.aggregate import AggregationFunction, RangedAggregate
from ezmsg.sigproc.math.log import Log
from ezmsg.sigproc.math.scale import Scale
from ezmsg.util.debuglog import DebugLog
from ezmsg.util.messages.chunker import ArrayChunker
from ezmsg.util.messages.modify import ModifyAxis
from ezmsg.util.terminate import TerminateOnTotal

from ezmsg.event.peak import ThresholdCrossing
from ezmsg.event.sparse import Densify
from ezmsg.event.util.simulate import generate_white_noise_with_events
from ezmsg.event.window import Window


def main(bin_duration: float = 0.05):
    fs = 30_000.0
    dur = 10.0
    n_chans = 128
    threshold = 2.5
    rate_range = (10, 100)
    chunk_dur = bin_duration / 2
    chunk_len = int(fs * chunk_dur)
    data = generate_white_noise_with_events(fs, dur, n_chans, rate_range, chunk_dur, threshold)
    n_chunks = int(dur / bin_duration)

    comps = {
        "SOURCE": ArrayChunker(data, chunk_len, fs=fs),
        "THRESH": ThresholdCrossing(threshold=threshold),
        "WIN": Window(
            axis="time",
            newaxis="win",
            window_dur=bin_duration,
            window_shift=bin_duration,
            zero_pad_until="none",
        ),
        "AGG": RangedAggregate(axis="time", bands=[(0, bin_duration)], operation=AggregationFunction.SUM),
        "SCALE": Scale(1 / bin_duration),
        "DENSE": Densify(),
        "AXIS": ModifyAxis(name_map={"time": None, "win": "time"}),
        "LOG10": Log(10, clip_zero=True),
        "SINK": DebugLog(),
        "TERM": TerminateOnTotal(n_chunks),
    }
    conns = (
        (comps["SOURCE"].OUTPUT_SIGNAL, comps["THRESH"].INPUT_SIGNAL),
        (comps["THRESH"].OUTPUT_SIGNAL, comps["WIN"].INPUT_SIGNAL),
        (comps["WIN"].OUTPUT_SIGNAL, comps["AGG"].INPUT_SIGNAL),
        (comps["AGG"].OUTPUT_SIGNAL, comps["SCALE"].INPUT_SIGNAL),
        (comps["SCALE"].OUTPUT_SIGNAL, comps["DENSE"].INPUT_SIGNAL),
        (comps["DENSE"].OUTPUT_SIGNAL, comps["AXIS"].INPUT_SIGNAL),
        (comps["AXIS"].OUTPUT_SIGNAL, comps["LOG10"].INPUT_SIGNAL),
        (comps["LOG10"].OUTPUT_SIGNAL, comps["SINK"].INPUT),
        (comps["SINK"].OUTPUT, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)


if __name__ == "__main__":
    typer.run(main)
