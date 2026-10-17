"""Chapter marks for the Hörplayer: ID3 title + CHAP frames, written after encoding.

Section start times come from the same timeline the track was rendered from:
cached clip lengths plus configured pauses, added up exactly like `render`
does. Nothing is decoded, and the audio bytes are never touched — the tag is
only placed in front of them.
"""

import wave
from dataclasses import dataclass
from pathlib import Path

from mutagen.id3 import CHAP, CTOC, CTOCFlags, ID3, ID3NoHeaderError, SYLT, TIT2

from germanaudio.assemble import Silence, TimelineItem
from germanaudio.audio import silence
from germanaudio.synth import clip_path

# A CTOC frame stores its child count in one byte, so one table lists at most 255 chapters.
_MAX_TOC_CHILDREN = 255


@dataclass(frozen=True)
class Chapter:
    title: str
    start_sample: int


@dataclass(frozen=True)
class Line:
    """One read-along entry: the text and the sample its spoken clip starts at."""

    text: str
    start_sample: int


def item_starts(
    timeline: list[TimelineItem], cache_dir: Path, engine_id: str, sample_rate: int
) -> tuple[list[int], int]:
    """Start sample of every timeline item, plus the track's total length in samples.

    Uses the identical arithmetic as `render` (clip frame counts, `silence()` rounding),
    reading only WAV headers, so the result matches the rendered audio sample for sample.
    """
    frames: dict[Path, int] = {}
    starts: list[int] = []
    position = 0
    for item in timeline:
        starts.append(position)
        if isinstance(item, Silence):
            position += len(silence(item.seconds, sample_rate))
            continue
        path = clip_path(cache_dir, item, engine_id)
        if path not in frames:
            with wave.open(str(path), "rb") as handle:
                frames[path] = handle.getnframes()
        position += frames[path]
    return starts, position


def mark_positions(
    timeline: list[TimelineItem],
    marks: dict[int, str],
    cache_dir: Path,
    engine_id: str,
    sample_rate: int,
) -> tuple[list[Chapter], int]:
    """Sample offset of every marked timeline item, plus the track's total length in samples."""
    starts, total = item_starts(timeline, cache_dir, engine_id, sample_rate)
    return [Chapter(marks[i], starts[i]) for i in sorted(marks)], total


def line_positions(starts: list[int], lines: dict[int, str]) -> list[Line]:
    """The read-along entries in time order, positioned like the chapters."""
    return [Line(lines[i], starts[i]) for i in sorted(lines)]


def write_tags(
    path: Path,
    title: str,
    chapters: list[Chapter],
    total_samples: int,
    sample_rate: int,
    lines: list[Line] | None = None,
) -> None:
    """Write the file title and one chapter per mark; each chapter ends where the next begins."""
    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = ID3()
    tags.delall("TIT2")
    tags.delall("CHAP")
    tags.delall("CTOC")
    tags.delall("SYLT")
    tags.add(TIT2(encoding=3, text=[title]))

    def ms(samples: int) -> int:
        return round(samples * 1000 / sample_rate)

    ids = []
    for i, chapter in enumerate(chapters):
        end = chapters[i + 1].start_sample if i + 1 < len(chapters) else total_samples
        element_id = f"chp{i}"
        tags.add(CHAP(element_id=element_id, start_time=ms(chapter.start_sample), end_time=ms(end),
                      start_offset=0xFFFFFFFF, end_offset=0xFFFFFFFF,
                      sub_frames=[TIT2(encoding=3, text=[chapter.title])]))
        ids.append(element_id)

    if ids:
        top = ids
        if len(ids) > _MAX_TOC_CHILDREN:
            top = []
            for n, first in enumerate(range(0, len(ids), _MAX_TOC_CHILDREN)):
                sub_id = f"toc{n}"
                tags.add(CTOC(element_id=sub_id, flags=CTOCFlags.ORDERED,
                              child_element_ids=ids[first:first + _MAX_TOC_CHILDREN],
                              sub_frames=[TIT2(encoding=3, text=[f"Teil {n + 1}"])]))
                top.append(sub_id)
        tags.add(CTOC(element_id="toc", flags=CTOCFlags.TOP_LEVEL | CTOCFlags.ORDERED,
                      child_element_ids=top, sub_frames=[TIT2(encoding=3, text=["Inhalt"])]))

    if lines:
        tags.add(SYLT(encoding=1, lang="deu", format=2, type=1, desc="",
                      text=[(line.text, ms(line.start_sample)) for line in lines]))

    tags.save(path, v2_version=3)
