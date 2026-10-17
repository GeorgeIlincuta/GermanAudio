"""Command-line entry point — wires the three stages together.

    python -m germanaudio vocab.tsv -o out/german.mp3
    python -m germanaudio vocab.tsv --check
    python -m germanaudio vocab.tsv --limit 8 -o out/sample.mp3
    python -m germanaudio story.txt --text -o out/story.mp3
"""

import argparse
import re
import sys
import wave
from pathlib import Path

from germanaudio.assemble import TimelineItem, build_timeline, encode_mp3, render, vocab_marks
from germanaudio.chapters import Chapter, item_starts, line_positions, write_tags
from germanaudio.config import (
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
        help="vocabulary file (tab- or pipe-separated), or a .txt file with --text",
    )
    parser.add_argument(
        "--text", action="store_true",
        help="read the input aloud as plain German text instead of a vocab list",
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
        help="title shown in the player (default: the story's first sentence, or the vocab file name)",
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


Plan = tuple[list[TimelineItem], dict[int, str], dict[int, str], str]  # timeline, chapter marks, read-along lines, file title


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
    return build_timeline(entries, voices, PauseConfig()), vocab_marks(entries), {}, title


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
    return text_timeline(paragraphs, voices.de, TextPauseConfig()), text_marks(paragraphs), text_lines(paragraphs), title


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    voices = VoiceConfig()

    plan = _text_timeline(args, voices) if args.text else _vocab_timeline(args, voices)
    if isinstance(plan, int):
        return plan
    timeline, marks, lines, title = plan

    if args.retag:
        engine_id, sample_rate = engine_identity()
        try:
            chapters_written, lines_written = retag_file(
                args.output, title, timeline, marks, lines, args.cache_dir, engine_id, sample_rate
            )
        except RetagError as error:
            print(f"retag failed: {error}", file=sys.stderr)
            return 1
        except (OSError, wave.Error) as error:
            print(f"retag failed: {error} — nothing changed", file=sys.stderr)
            return 1
        print(f"retagged {args.output}: title {title!r}, {chapters_written} chapters, {lines_written} text lines")
        return 0

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
        write_tags(args.output, title, chapters, total, engine.sample_rate, line_positions(starts, lines))
    except (OSError, wave.Error) as error:
        print(f"track written, but adding the title and chapters failed: {error}", file=sys.stderr)
        return 1
    print(f"  title {title!r}, {len(chapters)} chapters, {len(lines)} text lines")

    size_mb = args.output.stat().st_size / 1_000_000
    print(f"done: {args.output} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
