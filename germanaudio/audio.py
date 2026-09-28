"""WAV I/O and silence generation.

Uses the standard library's `wave` module rather than a dependency: the
pipeline only ever needs mono 16-bit PCM, which `wave` handles directly.

Samples are float32 in [-1.0, 1.0] everywhere in memory, and int16 on disk.
Keeping that boundary in one module means no other file deals in raw bytes.
"""

import wave
from pathlib import Path

import numpy as np

_INT16_SCALE = 32767.0


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(round(seconds * sample_rate)), dtype=np.float32)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clip before scaling: without this, a sample above 1.0 overflows int16
    # and wraps to a large negative value, which is audible as a loud click.
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * _INT16_SCALE).astype(np.int16)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1:
            raise ValueError(f"{path} is not mono; the pipeline only handles mono audio")
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    pcm = np.frombuffer(frames, dtype=np.int16)
    return (pcm.astype(np.float32) / _INT16_SCALE), sample_rate
