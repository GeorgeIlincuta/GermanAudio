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
    for one copy rather than two.
    """
    cache: dict[str, np.ndarray] = {}
    pieces: list[np.ndarray] = []

    for item in timeline:
        if isinstance(item, Silence):
            pieces.append(silence(item.seconds, sample_rate))
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
    return np.concatenate(pieces).astype(np.float32)


def encode_mp3(samples: np.ndarray, sample_rate: int, path: Path, bitrate: int) -> None:
    import lameenc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)  # 0 slowest/best, 9 fastest/worst

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    data = encoder.encode(pcm.tobytes())
    data += encoder.flush()

    path.write_bytes(bytes(data))
