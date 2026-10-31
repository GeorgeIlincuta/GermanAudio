"""Command-line entry point — wires the three stages together.

    python -m germanaudio vocab.tsv -o out/german.mp3
    python -m germanaudio vocab.tsv --check
    python -m germanaudio vocab.tsv --limit 8 -o out/sample.mp3
    python -m germanaudio story.txt --text -o out/story.mp3
    python -m germanaudio artikel/topic.md --article -o out/topic.mp3
    python -m germanaudio zucker.tsv -o out/zucker.mp3 --title "Zucker Vokabeln" --publish
"""

import argparse
import re
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

from germanaudio import library
from germanaudio.article import load_article
from germanaudio.assemble import Silence, TimelineItem, build_timeline, encode_mp3, render, vocab_marks
from germanaudio.chapters import ArticleInfo, Chapter, item_starts, line_positions, write_tags
from germanaudio.config import (
    ARTICLE_PART_GAP,
    CACHE_DIR,
    MP3_BITRATE,
    OUT_DIR,
    PauseConfig,
    TextPauseConfig,
    VoiceConfig,
)
from germanaudio.loader import LoadError, load_vocab
from germanaudio.retag import RetagError, retag_file
from germanaudio.reader import Heading, limit_sentences, load_text, story_title, text_lines, text_marks, text_timeline
from germanaudio.synth import ClipSpec, synthesize_clips


def engine_identity() -> tuple[str, int]:
    """The real engine's cache id and sample rate, without loading the model (retag only needs the cache keys)."""
    from germanaudio.config import SAMPLE_RATE
    from germanaudio.tts.supertonic import SupertonicEngine

    return SupertonicEngine.engine_id, SAMPLE_RATE


def build_engine():
    """Indirection so tests can substitute a fake without touching the model."""
    from germanaudio.tts.supertonic import SupertonicEngine

    return SupertonicEngine()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="germanaudio",
        description=(
            "Build a German vocabulary listening track from a TSV file, "
            "or read a German text file aloud with --text."
        ),
    )
    parser.add_argument(
        "input", type=Path,
        help="vocabulary file (tab- or pipe-separated), a .txt file with --text, or a .md file with --article",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--text", action="store_true",
        help="read the input aloud as plain German text instead of a vocab list",
    )
    mode.add_argument(
        "--article", action="store_true",
        help="build an Artikel track from a .md source file: its vocabulary, then the article",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=OUT_DIR / "german.mp3",
        help="output MP3 path (default: out/german.mp3)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=CACHE_DIR,
        help="where synthesized clips are cached (default: cache/audio)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="validate the input file and exit without synthesizing",
    )
    parser.add_argument(
        "--title", default=None,
        help="title shown in the player, without its number (default: the story's first sentence, or the vocab "
             "file name); the category number from the library is added in front: \"09 · Zucker Vokabeln\"",
    )
    parser.add_argument(
        "--library", type=Path, default=None,
        help="the Hörplayer library folder that numbers the titles and receives --publish "
             f"(default: {library.LIBRARY_ROOT})",
    )
    parser.add_argument(
        "--publish", action="store_true",
        help="after building or retagging, copy the track into the library folder and onto the phone (if connected)",
    )
    parser.add_argument(
        "--retag", action="store_true",
        help="rewrite only the tag (title, chapters, read-along text) of the existing output file from the cache; "
             "no synthesis, the audio is not touched",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only process the first N entries (sentences with --text), for previews",
    )
    return parser.parse_args(argv)


_WARNING_LINE = re.compile(r"^line (\d+):")


def _warning_applies(warning: str, retained_lines: set[int]) -> bool:
    """True unless the warning names a line that --limit dropped.

    Warnings are plain strings produced by the loader against the full
    file, before --limit narrows the entry list. A warning that doesn't
    match the "line N: ..." shape at all is kept rather than dropped, since
    silently swallowing an unrecognized warning is worse than showing one
    extra line.
    """
    match = _WARNING_LINE.match(warning)
    return match is None or int(match.group(1)) in retained_lines


def _format_duration(total_seconds: float) -> str:
    """"2h 13m 5s" past an hour, "13m 5s" otherwise.

    divmod(seconds, 60) alone reads fine for a sample run but prints
    "133m 5s" for a real ~500-entry track — technically correct, not
    something anyone wants to read.
    """
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def _report_load_error(path: Path, error: LoadError) -> None:
    print(f"{path}: input file has problems:", file=sys.stderr)
    for problem in error.problems:
        print(f"  {problem}", file=sys.stderr)


# The player's file type, written as the ID3 genre (TCON).
VOCAB_GENRE = "Wortschatz"
STORY_GENRE = "Geschichte"
ARTICLE_GENRE = "Artikel"


@dataclass(frozen=True)
class Plan:
    timeline: list[TimelineItem]
    marks: dict[int, str]  # timeline index -> chapter title
    lines: dict[int, str]  # timeline index -> read-along text
    title: str
    genre: str
    article: tuple[int, dict[str, str]] | None = None  # (timeline index of the article part, metadata)


def _vocab_timeline(args: argparse.Namespace, voices: VoiceConfig) -> Plan | int:
    """The vocab drill timeline with a chapter per entry, or an exit code if the run stops here."""
    try:
        result = load_vocab(args.input)
    except LoadError as error:
        _report_load_error(args.input, error)
        return 1

    entries = result.entries
    if args.limit is not None:
        entries = entries[: args.limit]

    retained_lines = {entry.line_number for entry in entries}
    for warning in result.warnings:
        if _warning_applies(warning, retained_lines):
            print(f"warning: {warning}", file=sys.stderr)

    print(f"loaded {len(entries)} entries from {args.input}")
    if args.check:
        return 0
    title = args.title or args.input.stem
    return Plan(build_timeline(entries, voices, PauseConfig()), vocab_marks(entries), {}, title, VOCAB_GENRE)


def _text_timeline(args: argparse.Namespace, voices: VoiceConfig) -> Plan | int:
    """The read-aloud timeline with a chapter per "#" heading, or an exit code if the run stops here."""
    try:
        paragraphs = load_text(args.input)
    except LoadError as error:
        _report_load_error(args.input, error)
        return 1

    if args.limit is not None:
        paragraphs = limit_sentences(paragraphs, args.limit)

    headings = sum(isinstance(block, Heading) for block in paragraphs)
    blocks = [block for block in paragraphs if not isinstance(block, Heading)]
    sentences = sum(len(block) for block in blocks)
    summary = f"loaded {sentences} sentences in {len(blocks)} paragraphs"
    if headings:
        summary += f" and {headings} headings"
    print(f"{summary} from {args.input}")
    if args.check:
        return 0
    title = args.title or story_title(paragraphs) or args.input.stem
    return Plan(text_timeline(paragraphs, voices.de, TextPauseConfig()), text_marks(paragraphs), text_lines(paragraphs),
                title, STORY_GENRE)


# Front-matter keys that become tag frames (the title goes to TIT2, "saved" stays in the source file).
_TAGGED_META = ("author", "published", "category", "source", "study")


def _article_timeline(args: argparse.Namespace, voices: VoiceConfig) -> Plan | int:
    """The vocab drill, a gap, then the article read aloud; or an exit code if the run stops here."""
    try:
        article = load_article(args.input)
    except LoadError as error:
        _report_load_error(args.input, error)
        return 1

    entries = article.entries
    paragraphs = article.paragraphs
    if args.limit is not None:
        entries = entries[: args.limit]
        paragraphs = limit_sentences(paragraphs, args.limit)
    for warning in article.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    headings = sum(isinstance(block, Heading) for block in paragraphs)
    blocks = [block for block in paragraphs if not isinstance(block, Heading)]
    sentences = sum(len(block) for block in blocks)
    print(f"loaded {len(entries)} entries and {sentences} sentences in {len(blocks)} paragraphs "
          f"and {headings} headings from {args.input}")
    if args.check:
        return 0
    if not entries:
        print(f"no vocabulary in {args.input}: fill the ## Wortschatz section first", file=sys.stderr)
        return 1

    vocab = build_timeline(entries, voices, PauseConfig())
    offset = len(vocab) + 1
    timeline = vocab + [Silence(ARTICLE_PART_GAP)] + text_timeline(paragraphs, voices.de, TextPauseConfig())
    marks = vocab_marks(entries) | {offset + i: title for i, title in text_marks(paragraphs).items()}
    lines = {offset + i: line for i, line in text_lines(paragraphs).items()}
    meta = {key: article.meta[key] for key in _TAGGED_META if key in article.meta}
    title = args.title or article.meta["title"]
    return Plan(timeline, marks, lines, title, ARTICLE_GENRE, (offset, meta))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    voices = VoiceConfig()

    if args.article:
        plan = _article_timeline(args, voices)
    elif args.text:
        plan = _text_timeline(args, voices)
    else:
        plan = _vocab_timeline(args, voices)
    if isinstance(plan, int):
        return plan
    if args.publish and args.limit is not None:
        print("--publish needs a full track: drop --limit", file=sys.stderr)
        return 2
    library_root = args.library or library.LIBRARY_ROOT
    timeline, marks, lines, genre = plan.timeline, plan.marks, plan.lines, plan.genre
    # a --limit preview isn't a library track, so it gets no number
    title = plan.title if args.limit is not None else library.numbered_title(
        plan.title, genre, args.output.name, library_root)

    if args.retag:
        engine_id, sample_rate = engine_identity()
        try:
            chapters_written, lines_written = retag_file(
                args.output, title, timeline, marks, lines, args.cache_dir, engine_id, sample_rate, genre,
                article=plan.article,
            )
        except RetagError as error:
            print(f"retag failed: {error}", file=sys.stderr)
            return 1
        except (OSError, wave.Error) as error:
            print(f"retag failed: {error} — nothing changed", file=sys.stderr)
            return 1
        print(f"retagged {args.output}: title {title!r}, genre {genre}, {chapters_written} chapters, {lines_written} text lines")
        return _publish(args, genre, title, library_root)

    # Checked now, not after an hour of synthesis: a bad output path should
    # fail in a second, before stage 2 even starts.
    args.output.parent.mkdir(parents=True, exist_ok=True)

    engine = build_engine()

    def progress(index: int, total: int, spec) -> None:
        print(f"\r  synthesizing {index}/{total}", end="", flush=True)

    print("synthesizing clips (cached clips are reused)...")
    clips = [item for item in timeline if isinstance(item, ClipSpec)]
    report = synthesize_clips(clips, engine, args.cache_dir, progress=progress)
    print()
    print(f"  {report.synthesized} new, {report.reused} reused from cache")

    if report.failures:
        print(f"{len(report.failures)} clips failed to synthesize:", file=sys.stderr)
        for spec, message in report.failures:
            print(f"  [{spec.language}] {spec.text!r}: {message}", file=sys.stderr)
        print("no track written; fix the above and re-run to resume", file=sys.stderr)
        return 1

    try:
        print("assembling track...")
        samples = render(timeline, args.cache_dir, engine.engine_id, engine.sample_rate)
    except (FileNotFoundError, ValueError, wave.Error) as error:
        # These three are exactly what a missing or corrupt cache entry
        # raises: FileNotFoundError when a clip was never synthesized,
        # ValueError on a sample-rate mismatch, wave.Error on a truncated
        # or non-WAV file. The recovery is the same for all three and is
        # not discoverable from the raw error message alone.
        print(f"failed to assemble the track: {error}", file=sys.stderr)
        print(
            f"this looks like a problem with a cached clip in {args.cache_dir}: "
            "delete the offending file (or the whole cache directory) and "
            "re-run to re-synthesize it",
            file=sys.stderr,
        )
        return 1
    except (MemoryError, OSError) as error:
        print(f"failed to assemble the track: {error}", file=sys.stderr)
        return 1

    print(f"  {_format_duration(len(samples) / engine.sample_rate)} of audio")

    try:
        print(f"encoding {args.output}...")
        encode_mp3(samples, engine.sample_rate, args.output, MP3_BITRATE)
    except (MemoryError, OSError) as error:
        print(f"failed to encode {args.output}: {error}", file=sys.stderr)
        return 1

    try:
        starts, total = item_starts(timeline, args.cache_dir, engine.engine_id, engine.sample_rate)
        chapters = [Chapter(marks[i], starts[i]) for i in sorted(marks)]
        article = ArticleInfo(starts[plan.article[0]], plan.article[1]) if plan.article is not None else None
        write_tags(args.output, title, chapters, total, engine.sample_rate, line_positions(starts, lines),
                   genre=genre, article=article)
    except (OSError, wave.Error) as error:
        print(f"track written, but adding the title and chapters failed: {error}", file=sys.stderr)
        return 1
    print(f"  title {title!r}, genre {genre}, {len(chapters)} chapters, {len(lines)} text lines")

    size_mb = args.output.stat().st_size / 1_000_000
    print(f"done: {args.output} ({size_mb:.1f} MB)")
    return _publish(args, genre, title, library_root)


def _publish(args: argparse.Namespace, genre: str, title: str, library_root: Path) -> int:
    if not args.publish:
        return 0
    ok, report = library.publish(args.output, genre, title, library_root, adb_path=library.ADB)
    for line in report:
        print(f"  {line}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
