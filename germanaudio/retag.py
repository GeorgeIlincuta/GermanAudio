"""Rewrite only the tag of an existing track — title, chapters, read-along text — from the clip cache.

Nothing is synthesized or encoded. The new tag is written to a copy, the audio after the tag is verified
byte for byte, and only then does the copy replace the original; on any problem the original is untouched.
"""

import hashlib
import os
import shutil
from pathlib import Path

from mutagen.mp3 import MP3

from germanaudio.assemble import TimelineItem
from germanaudio.chapters import Chapter, item_starts, line_positions, write_tags
from germanaudio.synth import ClipSpec, clip_path

# A rebuilt timeline may differ from the encoded audio by encoder padding, never by a sentence.
_MAX_LENGTH_DIFFERENCE_S = 0.5


class RetagError(Exception):
    pass


def audio_digest(path: Path) -> str:
    """SHA-256 of everything after the ID3v2 tag (the audio frames)."""
    data = Path(path).read_bytes()
    start = 0
    if data[:3] == b"ID3" and len(data) >= 10:
        size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
        start = 10 + size + (10 if data[5] & 0x10 else 0)
    return hashlib.sha256(data[start:]).hexdigest()


def missing_clip(timeline: list[TimelineItem], cache_dir: Path, engine_id: str) -> ClipSpec | None:
    for item in timeline:
        if isinstance(item, ClipSpec) and not clip_path(cache_dir, item, engine_id).is_file():
            return item
    return None


def retag_file(
    path: Path,
    title: str,
    timeline: list[TimelineItem],
    marks: dict[int, str],
    lines: dict[int, str],
    cache_dir: Path,
    engine_id: str,
    sample_rate: int,
) -> tuple[int, int]:
    """Retag [path] in place; returns (chapters, text lines). Raises RetagError and leaves the file untouched."""
    path = Path(path)
    if not path.is_file():
        raise RetagError(f"{path} does not exist")
    missing = missing_clip(timeline, cache_dir, engine_id)
    if missing is not None:
        raise RetagError(f"not in the cache: {missing.text!r} — nothing changed; rebuild the track instead")
    starts, total = item_starts(timeline, cache_dir, engine_id, sample_rate)
    audio_seconds = MP3(path).info.length
    if abs(audio_seconds - total / sample_rate) > _MAX_LENGTH_DIFFERENCE_S:
        raise RetagError(
            f"the text does not match the audio in {path} ({total / sample_rate:.1f} s vs {audio_seconds:.1f} s) "
            "— the text was probably edited after the build; rebuild the track instead"
        )
    chapters = [Chapter(marks[i], starts[i]) for i in sorted(marks)]
    before = audio_digest(path)
    temp = path.with_name(path.name + ".retag")
    shutil.copyfile(path, temp)
    try:
        write_tags(temp, title, chapters, total, sample_rate, line_positions(starts, lines))
        if audio_digest(temp) != before:
            raise RetagError("writing the tag changed the audio — nothing changed")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return len(chapters), len(lines)
