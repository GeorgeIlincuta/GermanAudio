"""Command-line entry point — wires the three stages together.

    python -m germanaudio vocab.tsv -o out/german.mp3
    python -m germanaudio vocab.tsv --check
    python -m germanaudio vocab.tsv --limit 8 -o out/sample.mp3
"""

import argparse
import sys
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
        help="only process the first N entries — useful for previewing settings",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        result = load_vocab(args.vocab)
    except LoadError as error:
        print(f"{args.vocab}: input file has problems:", file=sys.stderr)
        for problem in error.problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    entries = result.entries
    if args.limit is not None:
        entries = entries[: args.limit]

    print(f"loaded {len(entries)} entries from {args.vocab}")
    if args.check:
        return 0

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
        print("no track written — fix the above and re-run to resume", file=sys.stderr)
        return 1

    print("assembling track...")
    timeline = build_timeline(entries, voices, pauses)
    samples = render(timeline, args.cache_dir, engine.engine_id, engine.sample_rate)

    minutes, seconds = divmod(len(samples) / engine.sample_rate, 60)
    print(f"  {int(minutes)}m {int(seconds)}s of audio")

    print(f"encoding {args.output}...")
    encode_mp3(samples, engine.sample_rate, args.output, MP3_BITRATE)

    size_mb = args.output.stat().st_size / 1_000_000
    print(f"done: {args.output} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
