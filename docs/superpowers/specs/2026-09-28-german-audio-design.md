# GermanAudio — Design

**Date:** 2026-09-28
**Status:** Approved, pending implementation plan

## Goal

Turn a list of German vocabulary — word, English translation, and a German
example phrase containing the word — into a single long MP3 for passive
listening. No cloud services, no API keys, no per-use cost. Everything runs
locally.

Target scale: ~500 entries, producing roughly 2 hours of audio in one file.

## Non-goals

- Flashcard/Anki integration. Output is one audio file, not per-card media.
- Generating the vocabulary or the example phrases. The user produces the
  input file separately (see `prompts/generate-vocab.md`); this tool only
  reads it.
- Shuffling, spaced repetition, or any scheduling logic. Entries are spoken
  in file order.

## Input format

A plain text file with a header row and three columns.

```
german	english	phrase
das Fenster	the window	Bitte mach das Fenster zu, es ist kalt hier drinnen.
der Schlüssel	the key	Ich habe meinen Schlüssel zu Hause vergessen.
```

- **Encoding:** UTF-8. A byte-order mark, if present, is stripped.
- **Delimiter:** tab by default. The loader inspects the header line and
  accepts `|` (pipe) instead if no tab is present. This exists because the
  file is produced by copying model output out of a chat window, where tabs
  are occasionally mangled and pipes never are. Fields are stripped of
  surrounding whitespace after splitting.
- **Columns:** exactly three, in the order `german`, `english`, `phrase`.
  The header row is required and its names are validated.
- **Comments and blanks:** lines that are empty or begin with `#` are
  skipped.

No quoting rules, no escaping. Commas, quotation marks, and apostrophes in
the German text are ordinary characters. This is the reason for choosing a
tab/pipe delimiter over CSV.

## Output

One MP3: 44.1 kHz, mono, 64 kbps. At ~500 entries this is roughly 2 hours
and about 60 MB. Encoding uses `lameenc`, a pip-installable LAME binding,
so no external `ffmpeg` binary is required.

## Architecture

Three stages. Each has one job, a defined input, and a defined output.

```
vocab.tsv ──▶ [1] load & validate ──▶ list[Entry]
                                         │
                                         ▼
                        [2] synthesize ──▶ cache/audio/<sha1>.wav
                                         │   3 unique clips per entry
                                         ▼
                        [3] assemble ────▶ out/german.mp3
```

### Stage 1 — Load and validate

Reads the input file, produces `list[Entry]` where
`Entry = (german: str, english: str, phrase: str)`.

Validation collects **all** problems and reports them together with line
numbers, rather than aborting on the first one. A run against a 500-line
file should tell the user about every bad line in one pass.

Reported as errors (abort before any synthesis):

- Wrong column count on a line
- Any of the three fields empty after stripping
- Missing or misnamed header row
- File not valid UTF-8

Reported as warnings (run continues):

- Duplicate value in the `german` column
- A `phrase` that does not contain the `german` word as a substring
  (case-insensitive, and after stripping a leading article). German inflects,
  so this legitimately fires on correct data — it is a warning precisely
  because it cannot be trusted as an error.

### Stage 2 — Synthesize

For each entry, three distinct pieces of text need audio:

| Text | Language |
|---|---|
| `german` | `de` |
| `english` | `en` |
| `phrase` | `de` |

Each is synthesized once and stored at `cache/audio/<sha1>.wav`, where the
hash is over `(text, language, voice_id, engine_id)`. The doubled drill in
the output timeline reuses the same file — repetition costs track length,
not synthesis time.

Because the cache key covers the text and the voice, this gives resumability
and cheap edits for free: a crash at entry 400 resumes at 400 on the next
run, and correcting three rows in the input re-synthesizes three clips
rather than 1500.

A synthesis failure on an individual clip is recorded and the run continues.
At the end, if any clips failed, the stage reports them and stops before
assembly — a track with silent gaps where words should be is worse than no
track.

### Stage 3 — Assemble

Builds the timeline in memory as a sequence of (clip, silence) pairs,
concatenates at 44.1 kHz mono, and encodes to MP3.

Per-entry timeline:

| Position | Content | Language |
|---|---|---|
| 1 | word | de |
| — | pause | |
| 2 | translation | en |
| — | pause | |
| 3 | word (same clip as 1) | de |
| — | pause | |
| 4 | translation (same clip as 2) | en |
| — | pause | |
| 5 | example phrase | de |
| — | pause | |

Silence durations live in one config block:

```python
PAUSE_AFTER_WORD      = 1.5   # after first German word — recall gap
PAUSE_AFTER_TRANSLATION = 1.0
PAUSE_AFTER_REPEAT_WORD = 1.0
PAUSE_AFTER_REPEAT_TRANSLATION = 1.0
PAUSE_AFTER_PHRASE    = 2.0   # separates entries
```

At these values an entry runs roughly 15 seconds.

## TTS engine interface

The engine sits behind a single narrow protocol:

```python
class SynthesisEngine(Protocol):
    engine_id: str        # participates in the cache key
    sample_rate: int

    def synthesize(self, text: str, language: str, voice: str) -> np.ndarray:
        """Return mono float32 samples at self.sample_rate."""
```

The first implementation wraps **Supertonic 3** (ONNX Runtime, CPU, ~99M
parameters, 31 languages including `de` and `en`, 44.1 kHz output). Its
GitHub repository was archived 2026-09-09; the ONNX weights are pinned
locally and the archived state is acceptable for a local batch tool.

German pronunciation quality is the one thing that cannot be settled from
documentation, and it matters more here than in most TTS applications — a
learner internalizes whatever the model says. The build order therefore puts
a listening test first (see below). If the German is unacceptable, a second
implementation of this protocol (Piper has native German voices under MIT)
is one file, and the cache key's `engine_id` component means the two never
collide.

Voice selection is a config value per language. Changing it changes the
cache key, so a voice change rebuilds the clip set.

## Error handling

| Condition | Behavior |
|---|---|
| Input file missing / unreadable | Abort with the path and the OS error |
| Malformed input lines | Collect all, report with line numbers, abort |
| ONNX model files missing | Abort with the expected path and download instructions |
| Individual clip synthesis failure | Record, continue, report all at end, abort before assembly |
| Output directory unwritable | Checked before stage 2 starts, not after |
| Interrupt (Ctrl-C) during synthesis | Cache retains completed clips; next run resumes |

## Testing

TDD, with the TTS engine stubbed by a fake that returns deterministic
samples. Real audio quality is a human judgment made during the listening
test, not an automated assertion.

- **Loader:** valid file; tab and pipe delimiters; BOM; umlauts and ß;
  wrong column count; empty fields; missing header; comment and blank lines;
  duplicate detection; the substring warning firing and not firing.
- **Cache keys:** stable across runs for identical input; differ on text,
  language, voice, or engine change.
- **Assembly:** correct clip order and count per entry; the repeat positions
  reference the same cached file as the originals; silence durations match
  config; total duration equals the sum of parts.
- **Resume:** a run over a partially populated cache synthesizes only the
  missing clips.

## Project layout

```
GermanAudio/
  vocab.tsv                    user-supplied input
  prompts/
    generate-vocab.md          prompt for producing vocab.tsv
  germanaudio/
    __init__.py
    cli.py                     argument parsing, stage orchestration
    config.py                  pause durations, voices, paths, audio format
    models.py                  Entry dataclass
    loader.py                  stage 1
    synth.py                   stage 2 — cache orchestration
    assemble.py                stage 3 — timeline, concat, encode
    tts/
      __init__.py              SynthesisEngine protocol
      supertonic.py            Supertonic 3 / ONNX implementation
  tests/
    test_loader.py
    test_synth.py
    test_assemble.py
  models/                      ONNX weights (not in version control)
  cache/audio/                 synthesized clips (not in version control)
  out/                         generated MP3 (not in version control)
  requirements.txt
```

CLI:

```
python -m germanaudio vocab.tsv -o out/german.mp3
python -m germanaudio vocab.tsv --check        # stage 1 only
python -m germanaudio vocab.tsv --limit 8      # first N entries, for the listening test
```

Dependencies: `onnxruntime`, `numpy`, `lameenc`. Test-only: `pytest`.

## Build order

1. **Listening test.** Install Supertonic, pin the weights, synthesize about
   eight German words and phrases plus their English translations. The user
   listens and decides whether the German is good enough to build on. This
   gates everything after it.
2. Stage 1 — loader and validation.
3. Stage 2 — synthesis with the content-hash cache.
4. Stage 3 — timeline assembly and MP3 encoding.
5. End-to-end run against the real 500-entry file.

## Open risks

- **German pronunciation quality.** Addressed by step 1 of the build order;
  unresolvable before then. Mitigated structurally by the engine protocol.
- **Supertonic model licensing.** Sources disagree — the repository states
  sample code under MIT with model weights under OpenRAIL-M, while the
  Hugging Face listing is reported as MIT. Immaterial for personal use;
  needs resolving before any redistribution of generated audio.
