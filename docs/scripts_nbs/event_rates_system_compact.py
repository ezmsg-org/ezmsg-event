import ezmsg.core as ez
import typer
from ezmsg.sigproc.math.log import Log
from ezmsg.util.debuglog import DebugLog
from ezmsg.util.messages.chunker import ArrayChunker
from ezmsg.util.terminate import TerminateOnTotal

from ezmsg.event.peak import ThresholdCrossing
from ezmsg.event.rate import EventRate
from ezmsg.event.util.simulate import generate_white_noise_with_events


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
        "RATE": EventRate(bin_duration=bin_duration),
        "LOG10": Log(10, clip_zero=True),
        "SINK": DebugLog(),
        "TERM": TerminateOnTotal(n_chunks),
    }
    conns = (
        (comps["SOURCE"].OUTPUT_SIGNAL, comps["THRESH"].INPUT_SIGNAL),
        (comps["THRESH"].OUTPUT_SIGNAL, comps["RATE"].INPUT_SIGNAL),
        (comps["RATE"].OUTPUT_SIGNAL, comps["LOG10"].INPUT_SIGNAL),
        (comps["LOG10"].OUTPUT_SIGNAL, comps["SINK"].INPUT),
        (comps["SINK"].OUTPUT, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)


if __name__ == "__main__":
    typer.run(main)
