"""Command-line entry point — wires the three stages together.

    python -m germanaudio vocab.tsv -o out/german.mp3
    python -m germanaudio vocab.tsv --check
    python -m germanaudio vocab.tsv --limit 8 -o out/sample.mp3
"""

import argparse
import re
import sys
import wave
from pathlib import Path

from germanaudio.assemble import build_timeline, encode_mp3, render
from germanaudio.config import (
    CACHE_DIR,
    MP3_BITRATE,
    OUT_DIR,
    PauseConfig,
    VoiceConfig,
)
from germanaudio.loader import LoadError, load_vocab
from germanaudio.synth import synthesize_all


def build_engine():
    """Indirection so tests can substitute a fake without touching the model."""
    from germanaudio.tts.supertonic import SupertonicEngine

    return SupertonicEngine()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="germanaudio",
        description="Build a German vocabulary listening track from a TSV file.",
    )
    parser.add_argument("vocab", type=Path, help="input file (tab- or pipe-separated)")
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
        "--limit", type=int, default=None,
        help="only process the first N entries, useful for previewing settings",
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        result = load_vocab(args.vocab)
    except LoadError as error:
        print(f"{args.vocab}: input file has problems:", file=sys.stderr)
        for problem in error.problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    entries = result.entries
    if args.limit is not None:
        entries = entries[: args.limit]

    retained_lines = {entry.line_number for entry in entries}
    for warning in result.warnings:
        if _warning_applies(warning, retained_lines):
            print(f"warning: {warning}", file=sys.stderr)

    print(f"loaded {len(entries)} entries from {args.vocab}")
    if args.check:
        return 0

    # Checked now, not after an hour of synthesis: a bad output path should
    # fail in a second, before stage 2 even starts.
    args.output.parent.mkdir(parents=True, exist_ok=True)

    voices = VoiceConfig()
    pauses = PauseConfig()
    engine = build_engine()

    def progress(index: int, total: int, spec) -> None:
        print(f"\r  synthesizing {index}/{total}", end="", flush=True)

    print("synthesizing clips (cached clips are reused)...")
    report = synthesize_all(entries, engine, args.cache_dir, voices, progress=progress)
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
        timeline = build_timeline(entries, voices, pauses)
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

    size_mb = args.output.stat().st_size / 1_000_000
    print(f"done: {args.output} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
