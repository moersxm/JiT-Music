import os
from typing import Tuple
import numpy as np

try:
    import pretty_midi
except ImportError as e:
    raise RuntimeError("需要 pretty_midi：请先 `pip install pretty_midi`") from e


class MidiToPianoRoll:
    def __init__(self, fps: int = 16, min_pitch: int = 0, max_pitch: int = 127):
        self.fps = fps
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        assert 0 <= min_pitch <= max_pitch <= 127

    def midi_to_roll(self, midi_path: str) -> np.ndarray:
        pm = pretty_midi.PrettyMIDI(midi_path)
        if pm.get_end_time() <= 0:
            return np.zeros((32, self.max_pitch - self.min_pitch + 1), dtype=np.float32)
        pr = pm.get_piano_roll(fs=self.fps).T.astype(np.float32)  # [T, 128]
        pr = pr[:, self.min_pitch:self.max_pitch + 1]
        pr = np.clip(pr / 127.0, 0.0, 1.0).astype(np.float32)
        return pr


def roll_to_midi(roll: np.ndarray, fps: int = 16, min_pitch: int = 0, velocity: int = 100) -> "pretty_midi.PrettyMIDI":
    T, P = roll.shape
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    thr = 0.5
    on = (roll > thr).astype(np.int32)

    def t_sec(t): return t / float(fps)

    for p in range(P):
        pitch = min_pitch + p
        active = on[:, p]
        t = 0
        while t < T:
            if active[t]:
                s = t
                while t < T and active[t]:
                    t += 1
                e = t
                inst.notes.append(pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=int(pitch),
                    start=t_sec(s),
                    end=max(t_sec(e), t_sec(s) + 1.0 / fps)
                ))
            else:
                t += 1
    pm.instruments.append(inst)
    return pm


def save_roll_as_midi(roll: np.ndarray, path: str, fps: int = 16, min_pitch: int = 0, velocity: int = 100):
    pm = roll_to_midi(roll, fps=fps, min_pitch=min_pitch, velocity=velocity)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pm.write(path)