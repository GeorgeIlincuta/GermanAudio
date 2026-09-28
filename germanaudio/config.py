"""Tunable settings for the whole pipeline.

Every value a user might reasonably want to change lives here, so that
changing the feel of the track never means editing pipeline code.
"""

from dataclasses import dataclass
from pathlib import Path

# Audio format, fixed across the pipeline. Supertonic emits 44.1 kHz.
SAMPLE_RATE = 44100
MP3_BITRATE = 64  # kbps, mono — ~60 MB for two hours

CACHE_DIR = Path("cache/audio")
OUT_DIR = Path("out")


@dataclass(frozen=True)
class PauseConfig:
    """Silence after each slot of an entry, in seconds.

    The pause after the first German word is the recall gap — it is the one
    a learner actually uses, so it is the longest of the inter-slot pauses.
    """

    after_word: float = 1.5
    after_translation: float = 1.0
    after_repeat_word: float = 1.0
    after_repeat_translation: float = 1.0
    after_phrase: float = 2.0


@dataclass(frozen=True)
class VoiceConfig:
    """Supertonic preset voice name per language.

    Confirm these names against the installed model in the listening test —
    an unknown name fails at synthesis time, not at import time.
    """

    de: str = "M1"
    en: str = "M1"
