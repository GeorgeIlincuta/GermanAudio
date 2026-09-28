"""Stage 3 — lay clips and silence out in time, then encode.

`build_timeline` is deliberately pure: it decides the entire structure of the
track without touching a file. That makes the part most likely to need tuning
— the drill pattern and its pauses — the part that is trivial to test and
cheap to change.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from germanaudio.audio import read_wav, silence
from germanaudio.config import PauseConfig, VoiceConfig
from germanaudio.models import Entry
from germanaudio.synth import ClipSpec, clip_path, entry_clips

# Separate from audio.py's private _INT16_SCALE: that one belongs to the
# WAV cache boundary, this one belongs to the MP3 encode boundary. Same
# value, different modules, kept independent on purpose.
_INT16_SCALE = 32767.0

# lameenc.Encoder.encode() can be called repeatedly before flush(), so audio
# is fed in blocks rather than converting the whole track to int16 at once.
_ENCODE_BLOCK_SECONDS = 30


@dataclass(frozen=True)
class Silence:
    seconds: float


TimelineItem = ClipSpec | Silence


def build_timeline(
    entries: list[Entry],
    voices: VoiceConfig,
    pauses: PauseConfig,
) -> list[TimelineItem]:
    """German word, English, both again, then the example phrase.

    The repeat slots reuse the identical `ClipSpec` objects rather than
    rebuilding them, so they resolve to the same cached file.
    """
    timeline: list[TimelineItem] = []

    for entry in entries:
        word, translation, phrase = entry_clips(entry, voices)
        timeline.extend(
            [
                word,
                Silence(pauses.after_word),
                translation,
                Silence(pauses.after_translation),
                word,
                Silence(pauses.after_repeat_word),
                translation,
                Silence(pauses.after_repeat_translation),
                phrase,
                Silence(pauses.after_phrase),
            ]
        )

    return timeline


def render(
    timeline: list[TimelineItem],
    cache_dir: Path,
    engine_id: str,
    sample_rate: int,
) -> np.ndarray:
    """Concatenate the timeline into one waveform.

    Clips are read once each and reused, so the doubled drill costs memory
    for one copy rather than two. Silence gets the identical treatment: only
    five distinct pause durations exist, but a 500-entry timeline places
    roughly 2,500 silence items, so memoizing per (seconds, sample_rate)
    avoids thousands of redundant allocations of the same array contents.
    `np.concatenate` only reads its inputs, so reusing one array object at
    many positions is safe.
    """
    cache: dict[str, np.ndarray] = {}
    silence_cache: dict[tuple[float, int], np.ndarray] = {}
    pieces: list[np.ndarray] = []

    for item in timeline:
        if isinstance(item, Silence):
            silence_key = (item.seconds, sample_rate)
            if silence_key not in silence_cache:
                silence_cache[silence_key] = silence(item.seconds, sample_rate)
            pieces.append(silence_cache[silence_key])
            continue

        key = item.key(engine_id)
        if key not in cache:
            path = clip_path(cache_dir, item, engine_id)
            if not path.exists():
                raise FileNotFoundError(
                    f"no cached audio for {item.text!r} ({item.language}) at {path}"
                )
            samples, rate = read_wav(path)
            if rate != sample_rate:
                raise ValueError(
                    f"{path} is {rate} Hz but the track is {sample_rate} Hz"
                )
            cache[key] = samples
        pieces.append(cache[key])

    if not pieces:
        return np.zeros(0, dtype=np.float32)
    # `dtype=` casts as part of the concatenation itself. The previous
    # `.astype(np.float32)` ran on an already-float32 array purely to be
    # safe, but `.astype` copies unconditionally — at 500-entry scale that
    # redundant copy alone was measured at 1.4 GB.
    return np.concatenate(pieces, dtype=np.float32)


def encode_mp3(samples: np.ndarray, sample_rate: int, path: Path, bitrate: int) -> None:
    """Encode to MP3, feeding lameenc in blocks rather than all at once.

    Converting the entire track to int16 in one shot keeps the clipped
    float32 copy, the scaled product, the int16 array, and its `tobytes()`
    copy all live simultaneously — at 500-entry scale, on the order of
    4 GB combined. `lameenc.Encoder.encode()` may be called repeatedly
    before `flush()`, so each ~30-second block is converted and encoded on
    its own, keeping the live working set at a few MB regardless of the
    track's total length.
    """
    import lameenc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)  # 0 slowest/best, 9 fastest/worst

    block_frames = _ENCODE_BLOCK_SECONDS * sample_rate
    chunks: list[bytes] = []
    for start in range(0, len(samples), block_frames):
        block = samples[start : start + block_frames]
        pcm = (np.clip(block, -1.0, 1.0) * _INT16_SCALE).astype(np.int16)
        chunks.append(encoder.encode(pcm.tobytes()))
    chunks.append(encoder.flush())

    path.write_bytes(b"".join(chunks))
