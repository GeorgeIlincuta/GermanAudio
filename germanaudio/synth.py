"""Stage 2 — turn entries into cached audio clips.

Each distinct (text, language, voice, engine) combination is synthesized once
and stored under a hash of exactly those four things. That single decision
buys three properties at once: repeated words across entries cost nothing,
an interrupted run resumes where it stopped, and correcting three rows in the
input re-synthesizes three clips rather than the whole file.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from germanaudio.audio import write_wav
from germanaudio.config import VoiceConfig
from germanaudio.models import Entry
from germanaudio.tts import SynthesisEngine


@dataclass(frozen=True)
class ClipSpec:
    text: str
    language: str
    voice: str

    def key(self, engine_id: str) -> str:
        # The separator must be a character that cannot occur in any field,
        # so that ("ab", "c") and ("a", "bc") cannot collide.
        payload = "\x00".join([engine_id, self.language, self.voice, self.text])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SynthesisReport:
    synthesized: int = 0
    reused: int = 0
    failures: list[tuple[ClipSpec, str]] = field(default_factory=list)


def entry_clips(entry: Entry, voices: VoiceConfig) -> tuple[ClipSpec, ClipSpec, ClipSpec]:
    """The three unique clips an entry needs, in slot order."""
    return (
        ClipSpec(entry.german, "de", voices.de),
        ClipSpec(entry.english, "en", voices.en),
        ClipSpec(entry.phrase, "de", voices.de),
    )


def required_clips(entries: list[Entry], voices: VoiceConfig) -> list[ClipSpec]:
    """Every distinct clip across all entries, in first-seen order."""
    seen: dict[ClipSpec, None] = {}
    for entry in entries:
        for spec in entry_clips(entry, voices):
            seen.setdefault(spec, None)
    return list(seen)


def clip_path(cache_dir: Path, spec: ClipSpec, engine_id: str) -> Path:
    return Path(cache_dir) / f"{spec.key(engine_id)}.wav"


def synthesize_all(
    entries: list[Entry],
    engine: SynthesisEngine,
    cache_dir: Path,
    voices: VoiceConfig,
    progress: Callable[[int, int, ClipSpec], None] | None = None,
) -> SynthesisReport:
    return synthesize_clips(required_clips(entries, voices), engine, cache_dir, progress)


def synthesize_clips(
    specs: list[ClipSpec],
    engine: SynthesisEngine,
    cache_dir: Path,
    progress: Callable[[int, int, ClipSpec], None] | None = None,
) -> SynthesisReport:
    """Synthesize any clip not already cached. Duplicate specs cost nothing."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    specs = list(dict.fromkeys(specs))
    synthesized = 0
    reused = 0
    failures: list[tuple[ClipSpec, str]] = []

    for index, spec in enumerate(specs, start=1):
        if progress is not None:
            progress(index, len(specs), spec)

        path = clip_path(cache_dir, spec, engine.engine_id)
        if path.exists():
            reused += 1
            continue

        try:
            samples = engine.synthesize(spec.text, spec.language, spec.voice)
            write_wav(path, samples, engine.sample_rate)
        except Exception as error:  # noqa: BLE001 — one bad clip must not end the run
            # Remove anything half-written, so a later run retries this clip
            # rather than assembling a truncated one into the track.
            path.unlink(missing_ok=True)
            failures.append((spec, str(error)))
            continue

        synthesized += 1

    return SynthesisReport(synthesized=synthesized, reused=reused, failures=failures)
