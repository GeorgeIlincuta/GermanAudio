# GermanAudio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a three-column vocabulary file (German word, English translation, German example phrase) into one long MP3 for passive listening, entirely offline and at zero per-use cost.

**Architecture:** Three stages with files between them. Stage 1 loads and validates the input into `list[Entry]`. Stage 2 synthesizes three unique audio clips per entry through a narrow `SynthesisEngine` protocol, caching each by a content hash so runs resume and edits are cheap. Stage 3 builds a pure timeline of clips and silences, renders it to samples, and encodes MP3. The TTS engine is the only replaceable part and is isolated behind one method.

**Tech Stack:** Python 3.10+, `supertonic` (ONNX Runtime, CPU), `numpy`, `lameenc`, `pytest`. Standard-library `wave` for WAV I/O.

**Spec:** `docs/superpowers/specs/2026-09-28-german-audio-design.md`

## Global Constraints

- **Python 3.10 or newer.** The code uses `X | Y` union syntax in annotations.
- **No network access at build time.** The only network use is the one-off model download in Task 1. Stages 1–3 run fully offline.
- **No external binaries.** MP3 encoding uses `lameenc` (pip). Never shell out to `ffmpeg`.
- **All file I/O is UTF-8 explicit.** Every `open()` on a text file passes `encoding="utf-8"`. The platform is Windows, where the default encoding is not UTF-8 and German umlauts will corrupt silently without this.
- **Audio format throughout:** 44100 Hz, mono, `float32` in memory with samples in `[-1.0, 1.0]`, `int16` on disk.
- **Language codes are exactly `"de"` and `"en"`** — Supertonic's two-letter codes.
- **Tests never call the real TTS engine.** They use `FakeEngine` from `tests/conftest.py`.

---

### Task 1: Project scaffolding and Supertonic listening gate

This task ends with a human decision, not just passing tests. Do not start Task 2 until the user has listened to the samples and approved the German quality.

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `germanaudio/__init__.py`
- Create: `germanaudio/config.py`
- Create: `scripts/listening_test.py`

**Interfaces:**
- Consumes: nothing
- Produces: `germanaudio/config.py` exporting `SAMPLE_RATE: int`, `MP3_BITRATE: int`, `CACHE_DIR: Path`, `OUT_DIR: Path`, and the frozen dataclasses `PauseConfig` and `VoiceConfig` described below. Every later task imports from here.

- [ ] **Step 1: Initialize the repository**

This directory is not yet a git repository, and every task below ends in a commit.

```bash
git init
git branch -M main
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.pytest_cache/
.venv/
venv/

# Generated and downloaded artifacts — never committed
cache/
out/
assets/
results/
vocab.tsv
```

- [ ] **Step 3: Write `requirements.txt`**

```
supertonic
numpy
lameenc
pytest
```

- [ ] **Step 4: Install dependencies**

```bash
python -m pip install -r requirements.txt
```

Expected: all four install cleanly. `supertonic` pulls in `onnxruntime`.

- [ ] **Step 5: Write `germanaudio/__init__.py`**

```python
"""Turn a German vocabulary file into a long listening track."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Write `germanaudio/config.py`**

```python
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
```

- [ ] **Step 7: Write the listening-test script**

This script exists to answer one question that no automated test can: does this model's German sound right? It also prints the facts Task 4 needs — the real voice names, the sample rate, and the dtype `synthesize` returns.

Create `scripts/listening_test.py`:

```python
"""Synthesize a handful of samples so a human can judge German quality.

Run this before building anything else. If the German is wrong, the engine
gets swapped before the rest of the pipeline is written.

    python scripts/listening_test.py
"""

from pathlib import Path

import numpy as np
from supertonic import TTS

OUT = Path("out/listening_test")

SAMPLES = [
    ("de", "das Fenster"),
    ("de", "der Schlüssel"),
    ("de", "die Möglichkeit"),
    ("de", "Bitte mach das Fenster zu, es ist kalt hier drinnen."),
    ("de", "Ich habe meinen Schlüssel schon wieder zu Hause vergessen."),
    ("de", "Er hat sich über die schlechte Nachricht sehr geärgert."),
    ("en", "the window"),
    ("en", "the key"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    tts = TTS(auto_download=True)
    style = tts.get_voice_style(voice_name="M1")

    for index, (lang, text) in enumerate(SAMPLES, start=1):
        wav, duration = tts.synthesize(text, voice_style=style, lang=lang)

        if index == 1:
            # Record what the engine actually hands back — Task 4's adapter
            # has to normalize this to float32 mono.
            array = np.asarray(wav)
            print("--- engine facts (note these down) ---")
            print(f"  type returned : {type(wav)}")
            print(f"  dtype         : {array.dtype}")
            print(f"  shape         : {array.shape}")
            print(f"  min / max     : {array.min():.4f} / {array.max():.4f}")
            print(f"  duration (s)  : {duration}")
            print("--------------------------------------")

        path = OUT / f"{index:02d}_{lang}.wav"
        tts.save_audio(wav, str(path))
        print(f"{path}  ({duration:.2f}s)  {text}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run the listening test**

```bash
python scripts/listening_test.py
```

Expected: the model downloads on first run (this takes a few minutes), then eight WAV files appear in `out/listening_test/` and the engine-facts block prints.

If `get_voice_style(voice_name="M1")` raises, the preset name is wrong for this build. Inspect what the model ships:

```bash
python -c "from supertonic import TTS; t = TTS(auto_download=True); print([m for m in dir(t) if 'voice' in m.lower() or 'style' in m.lower()])"
```

Use whatever listing method that reveals to find a valid name, then update both `SAMPLES` usage and `VoiceConfig` defaults in `config.py`.

- [ ] **Step 9: Record the engine facts in `config.py`**

If the printed sample rate differs from 44100, or the voice names differ from `M1`, update `SAMPLE_RATE` and `VoiceConfig` now. Later tasks read these as truth.

- [ ] **Step 10: STOP — human listening gate**

Play all eight files. Report to the user:

- Whether German umlauts and `ß` are pronounced correctly
- Whether word stress in the multi-word phrases sounds native
- Whether the `ch` sound (in `Möglichkeit`, `ärgern`) is right
- Whether final consonants are devoiced correctly (`Tag` should end in a `k` sound)

**Ask the user to listen and approve before continuing.** If the German is not good enough, stop here — the fix is a different engine behind the same protocol (Task 4), not more code on top of this one.

- [ ] **Step 11: Commit**

```bash
git add .gitignore requirements.txt germanaudio/__init__.py germanaudio/config.py scripts/listening_test.py
git commit -m "feat: project scaffolding and Supertonic listening test"
```

---

### Task 2: Entry model and input loader

**Files:**
- Create: `germanaudio/models.py`
- Create: `germanaudio/loader.py`
- Create: `tests/__init__.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `germanaudio.models.Entry` — frozen dataclass with fields `german: str`, `english: str`, `phrase: str`, `line_number: int`
  - `germanaudio.loader.LoadError(Exception)` — carries `problems: list[str]`
  - `germanaudio.loader.LoadResult` — frozen dataclass with `entries: list[Entry]`, `warnings: list[str]`
  - `germanaudio.loader.load_vocab(path: Path) -> LoadResult` — raises `LoadError`

- [ ] **Step 1: Write the failing tests**

Create `tests/__init__.py` (empty file), then `tests/test_loader.py`:

```python
from pathlib import Path

import pytest

from germanaudio.loader import LoadError, load_vocab


def write(tmp_path: Path, content: str, name: str = "vocab.tsv") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_tab_separated_entries(tmp_path):
    path = write(
        tmp_path,
        "german\tenglish\tphrase\n"
        "das Fenster\tthe window\tBitte mach das Fenster zu, es ist kalt.\n"
        "der Schlüssel\tthe key\tIch habe meinen Schlüssel vergessen.\n",
    )

    result = load_vocab(path)

    assert len(result.entries) == 2
    assert result.entries[0].german == "das Fenster"
    assert result.entries[0].english == "the window"
    assert result.entries[0].phrase == "Bitte mach das Fenster zu, es ist kalt."
    assert result.entries[0].line_number == 2


def test_loads_pipe_separated_entries(tmp_path):
    path = write(
        tmp_path,
        "german | english | phrase\n"
        "das Fenster | the window | Bitte mach das Fenster zu, es ist kalt.\n",
    )

    result = load_vocab(path)

    assert len(result.entries) == 1
    assert result.entries[0].german == "das Fenster"
    assert result.entries[0].phrase == "Bitte mach das Fenster zu, es ist kalt."


def test_commas_inside_phrases_are_not_separators(tmp_path):
    path = write(
        tmp_path,
        "german\tenglish\tphrase\n"
        "der Hund\tthe dog\tDer Hund, der dort liegt, gehört meiner Schwester.\n",
    )

    result = load_vocab(path)

    assert result.entries[0].phrase == "Der Hund, der dort liegt, gehört meiner Schwester."


def test_strips_byte_order_mark(tmp_path):
    path = tmp_path / "vocab.tsv"
    path.write_text(
        "german\tenglish\tphrase\nder Tag\tthe day\tHeute ist ein schöner Tag gewesen.\n",
        encoding="utf-8-sig",
    )

    result = load_vocab(path)

    assert result.entries[0].german == "der Tag"


def test_preserves_umlauts_and_eszett(tmp_path):
    path = write(
        tmp_path,
        "german\tenglish\tphrase\n"
        "die Straße\tthe street\tDie Straße war größer als erwartet.\n",
    )

    result = load_vocab(path)

    assert result.entries[0].german == "die Straße"
    assert "größer" in result.entries[0].phrase


def test_skips_blank_and_comment_lines(tmp_path):
    path = write(
        tmp_path,
        "german\tenglish\tphrase\n"
        "# batch one\n"
        "\n"
        "das Buch\tthe book\tIch lese jeden Abend ein gutes Buch.\n",
    )

    result = load_vocab(path)

    assert len(result.entries) == 1
    assert result.entries[0].line_number == 4


def test_rejects_wrong_column_count_with_line_number(tmp_path):
    path = write(
        tmp_path,
        "german\tenglish\tphrase\n"
        "das Buch\tthe book\tIch lese ein Buch.\n"
        "der Stuhl\tthe chair\n",
    )

    with pytest.raises(LoadError) as excinfo:
        load_vocab(path)

    assert any("line 3" in problem for problem in excinfo.value.problems)


def test_rejects_empty_field(tmp_path):
    path = write(
        tmp_path,
        "german\tenglish\tphrase\nder Stuhl\t\tIch sitze auf dem Stuhl.\n",
    )

    with pytest.raises(LoadError) as excinfo:
        load_vocab(path)

    assert any("line 2" in problem for problem in excinfo.value.problems)


def test_reports_every_bad_line_not_just_the_first(tmp_path):
    path = write(
        tmp_path,
        "german\tenglish\tphrase\n"
        "bad one\ttwo\n"
        "das Buch\tthe book\tIch lese ein Buch.\n"
        "bad two\n"
        "\t\t\n",
    )

    with pytest.raises(LoadError) as excinfo:
        load_vocab(path)

    assert len(excinfo.value.problems) == 3


def test_rejects_missing_header(tmp_path):
    path = write(
        tmp_path,
        "das Fenster\tthe window\tBitte mach das Fenster zu.\n",
    )

    with pytest.raises(LoadError) as excinfo:
        load_vocab(path)

    assert any("header" in problem.lower() for problem in excinfo.value.problems)


def test_warns_on_duplicate_german_word(tmp_path):
    path = write(
        tmp_path,
        "german\tenglish\tphrase\n"
        "das Buch\tthe book\tIch lese ein Buch.\n"
        "das Buch\tthe book\tDas Buch liegt auf dem Tisch.\n",
    )

    result = load_vocab(path)

    assert len(result.entries) == 2
    assert any("duplicate" in warning.lower() for warning in result.warnings)


def test_warns_when_phrase_omits_the_word(tmp_path):
    path = write(
        tmp_path,
        "german\tenglish\tphrase\nder Stuhl\tthe chair\tIch sitze am Tisch.\n",
    )

    result = load_vocab(path)

    assert len(result.entries) == 1
    assert any("der Stuhl" in warning for warning in result.warnings)


def test_no_warning_when_word_appears_inflected(tmp_path):
    path = write(
        tmp_path,
        "german\tenglish\tphrase\n"
        "der Schlüssel\tthe key\tIch habe meinen Schlüssel verloren.\n",
    )

    result = load_vocab(path)

    assert result.warnings == []


def test_rejects_missing_file(tmp_path):
    with pytest.raises(LoadError):
        load_vocab(tmp_path / "does-not-exist.tsv")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_loader.py -v
```

Expected: every test fails with `ModuleNotFoundError: No module named 'germanaudio.loader'`.

- [ ] **Step 3: Write `germanaudio/models.py`**

```python
"""Core data types shared across the pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    """One vocabulary item: a word, its translation, and an example.

    `line_number` is 1-based and refers to the source file, so that any
    problem found downstream can be pointed back at a line the user can edit.
    """

    german: str
    english: str
    phrase: str
    line_number: int
```

- [ ] **Step 4: Write `germanaudio/loader.py`**

```python
"""Stage 1 — read the vocabulary file and validate it.

Validation collects every problem before raising, rather than failing on the
first one. Against a 500-line file, one run should tell the user about every
bad line, not send them round the loop 500 times.
"""

from dataclasses import dataclass, field
from pathlib import Path

from germanaudio.models import Entry

EXPECTED_COLUMNS = ("german", "english", "phrase")

# German articles, stripped before checking whether a phrase contains its word.
ARTICLES = ("der ", "die ", "das ")


class LoadError(Exception):
    """Raised when the input file cannot be used. Carries every problem found."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("\n".join(problems))


@dataclass(frozen=True)
class LoadResult:
    entries: list[Entry]
    warnings: list[str] = field(default_factory=list)


def _detect_delimiter(header: str) -> str:
    """Tab is the documented format; pipe is the fallback.

    Copying tab-separated text out of a chat window sometimes converts tabs
    to spaces, and that failure is silent. Pipes survive copy/paste and never
    occur inside German text, so accepting both costs little and saves a
    confusing class of bug reports.
    """
    return "\t" if "\t" in header else "|"


def _contains_word(phrase: str, german: str) -> bool:
    word = german.lower()
    for article in ARTICLES:
        if word.startswith(article):
            word = word[len(article):]
            break
    return word in phrase.lower()


def load_vocab(path: Path) -> LoadResult:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise LoadError([f"cannot read {path}: {error}"]) from error

    lines = text.splitlines()
    if not lines:
        raise LoadError([f"{path} is empty"])

    delimiter = _detect_delimiter(lines[0])
    header = [cell.strip().lower() for cell in lines[0].split(delimiter)]
    if tuple(header) != EXPECTED_COLUMNS:
        raise LoadError(
            [
                "line 1: header must be exactly "
                f"{delimiter.join(EXPECTED_COLUMNS)!r}, found {lines[0]!r}"
            ]
        )

    entries: list[Entry] = []
    problems: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for offset, raw in enumerate(lines[1:], start=2):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue

        cells = [cell.strip() for cell in raw.split(delimiter)]
        if len(cells) != 3:
            problems.append(f"line {offset}: expected 3 columns, found {len(cells)}")
            continue
        if not all(cells):
            problems.append(f"line {offset}: every column must be filled")
            continue

        german, english, phrase = cells
        if german.lower() in seen:
            warnings.append(f"line {offset}: duplicate entry for {german!r}")
        seen.add(german.lower())

        if not _contains_word(phrase, german):
            warnings.append(
                f"line {offset}: phrase may not contain {german!r} — check it reads naturally"
            )

        entries.append(Entry(german, english, phrase, offset))

    if problems:
        raise LoadError(problems)
    if not entries:
        raise LoadError([f"{path} contains a header but no entries"])

    return LoadResult(entries=entries, warnings=warnings)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest tests/test_loader.py -v
```

Expected: all 14 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add germanaudio/models.py germanaudio/loader.py tests/__init__.py tests/test_loader.py
git commit -m "feat: vocabulary file loader with full-file validation"
```

---

### Task 3: WAV read and write

**Files:**
- Create: `germanaudio/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: `germanaudio.config.SAMPLE_RATE`
- Produces:
  - `germanaudio.audio.write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None` — takes float32 mono in `[-1, 1]`, writes 16-bit PCM
  - `germanaudio.audio.read_wav(path: Path) -> tuple[np.ndarray, int]` — returns float32 mono and the sample rate
  - `germanaudio.audio.silence(seconds: float, sample_rate: int) -> np.ndarray` — float32 zeros

- [ ] **Step 1: Write the failing tests**

Create `tests/test_audio.py`:

```python
import numpy as np
import pytest

from germanaudio.audio import read_wav, silence, write_wav


def test_round_trip_preserves_samples(tmp_path):
    original = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    path = tmp_path / "clip.wav"

    write_wav(path, original, 44100)
    restored, rate = read_wav(path)

    assert rate == 44100
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, original, atol=1e-4)


def test_round_trip_preserves_full_scale_extremes(tmp_path):
    original = np.array([1.0, -1.0], dtype=np.float32)
    path = tmp_path / "clip.wav"

    write_wav(path, original, 44100)
    restored, _ = read_wav(path)

    np.testing.assert_allclose(restored, original, atol=1e-4)


def test_clips_out_of_range_samples(tmp_path):
    """Values beyond +/-1 must clamp, not wrap around into loud noise."""
    original = np.array([2.0, -2.0], dtype=np.float32)
    path = tmp_path / "clip.wav"

    write_wav(path, original, 44100)
    restored, _ = read_wav(path)

    assert restored.max() <= 1.0
    assert restored.min() >= -1.0


def test_silence_has_correct_length_and_is_quiet():
    samples = silence(1.5, 44100)

    assert len(samples) == 66150
    assert samples.dtype == np.float32
    assert np.all(samples == 0.0)


def test_silence_of_zero_seconds_is_empty():
    assert len(silence(0.0, 44100)) == 0


def test_read_wav_rejects_stereo(tmp_path):
    import wave

    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00\x00\x00")

    with pytest.raises(ValueError, match="mono"):
        read_wav(path)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_audio.py -v
```

Expected: all fail with `ModuleNotFoundError: No module named 'germanaudio.audio'`.

- [ ] **Step 3: Write `germanaudio/audio.py`**

```python
"""WAV I/O and silence generation.

Uses the standard library's `wave` module rather than a dependency: the
pipeline only ever needs mono 16-bit PCM, which `wave` handles directly.

Samples are float32 in [-1.0, 1.0] everywhere in memory, and int16 on disk.
Keeping that boundary in one module means no other file deals in raw bytes.
"""

import wave
from pathlib import Path

import numpy as np

_INT16_SCALE = 32767.0


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(round(seconds * sample_rate)), dtype=np.float32)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clip before scaling: without this, a sample above 1.0 overflows int16
    # and wraps to a large negative value, which is audible as a loud click.
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * _INT16_SCALE).astype(np.int16)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1:
            raise ValueError(f"{path} is not mono; the pipeline only handles mono audio")
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    pcm = np.frombuffer(frames, dtype=np.int16)
    return (pcm.astype(np.float32) / _INT16_SCALE), sample_rate
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_audio.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add germanaudio/audio.py tests/test_audio.py
git commit -m "feat: mono WAV read/write and silence generation"
```

---

### Task 4: Synthesis engine protocol and Supertonic adapter

**Files:**
- Create: `germanaudio/tts/__init__.py`
- Create: `germanaudio/tts/supertonic.py`
- Create: `tests/conftest.py`
- Test: `tests/test_supertonic.py`

**Interfaces:**
- Consumes: `germanaudio.config.SAMPLE_RATE`
- Produces:
  - `germanaudio.tts.SynthesisEngine` — `Protocol` with attributes `engine_id: str`, `sample_rate: int` and method `synthesize(text: str, language: str, voice: str) -> np.ndarray`
  - `germanaudio.tts.supertonic.SupertonicEngine` — concrete implementation, `engine_id = "supertonic-3"`
  - `tests.conftest.FakeEngine` — pytest fixture-friendly stub used by Tasks 5 and 6

- [ ] **Step 1: Write the failing tests**

Create `tests/conftest.py`. Every later task's tests use this rather than the real model — synthesis is slow, needs downloaded weights, and its output is judged by ear, not by assertion.

```python
"""Shared test doubles."""

import hashlib

import numpy as np
import pytest


class FakeEngine:
    """Deterministic stand-in for a real TTS engine.

    Returns a distinct, reproducible tone per input so that tests can assert
    which clip landed where without listening to anything. Length scales with
    text length, which makes duration assertions meaningful.
    """

    engine_id = "fake"
    sample_rate = 44100

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def synthesize(self, text: str, language: str, voice: str) -> np.ndarray:
        self.calls.append((text, language, voice))

        digest = hashlib.sha1(text.encode("utf-8")).digest()
        frequency = 200 + digest[0]  # 200-455 Hz, stable per text
        duration = 0.05 * len(text)

        t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()
```

Then `tests/test_supertonic.py`. These tests exercise the adapter's normalization logic against a stubbed `TTS` class — they never download or run the real model.

```python
import numpy as np
import pytest

from germanaudio.tts.supertonic import SupertonicEngine


class StubTTS:
    """Mimics supertonic.TTS closely enough to test the adapter's conversions."""

    def __init__(self, payload, **kwargs):
        self.payload = payload
        self.styles_requested: list[str] = []

    def get_voice_style(self, voice_name: str):
        self.styles_requested.append(voice_name)
        return f"style::{voice_name}"

    def synthesize(self, text, voice_style, lang):
        return self.payload, 1.0


def engine_with(payload) -> SupertonicEngine:
    return SupertonicEngine(tts=StubTTS(payload))


def test_returns_float32():
    engine = engine_with(np.array([0.1, -0.2], dtype=np.float64))

    result = engine.synthesize("das Fenster", "de", "M1")

    assert result.dtype == np.float32


def test_converts_int16_output_to_normalized_float():
    engine = engine_with(np.array([32767, -32768], dtype=np.int16))

    result = engine.synthesize("das Fenster", "de", "M1")

    assert result.dtype == np.float32
    assert result.max() <= 1.0
    assert result.min() >= -1.0
    assert result[0] == pytest.approx(1.0, abs=1e-3)


def test_flattens_stereo_to_mono():
    stereo = np.array([[0.4, 0.2], [0.6, 0.4]], dtype=np.float32)
    engine = engine_with(stereo)

    result = engine.synthesize("das Fenster", "de", "M1")

    assert result.ndim == 1
    assert len(result) == 2
    assert result[0] == pytest.approx(0.3)


def test_accepts_plain_list_output():
    engine = engine_with([0.1, 0.2, 0.3])

    result = engine.synthesize("das Fenster", "de", "M1")

    assert result.dtype == np.float32
    assert len(result) == 3


def test_caches_voice_styles_across_calls():
    """Building a style is not free; the same voice must not rebuild it."""
    stub = StubTTS(np.array([0.1], dtype=np.float32))
    engine = SupertonicEngine(tts=stub)

    engine.synthesize("eins", "de", "M1")
    engine.synthesize("zwei", "de", "M1")
    engine.synthesize("three", "en", "M1")

    assert stub.styles_requested == ["M1"]


def test_rejects_empty_text():
    engine = engine_with(np.array([0.1], dtype=np.float32))

    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", "de", "M1")


def test_engine_id_is_stable():
    assert SupertonicEngine(tts=StubTTS(np.array([0.1]))).engine_id == "supertonic-3"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_supertonic.py -v
```

Expected: all fail with `ModuleNotFoundError: No module named 'germanaudio.tts'`.

- [ ] **Step 3: Write `germanaudio/tts/__init__.py`**

```python
"""The one replaceable part of the pipeline.

Everything downstream talks to this protocol and nothing else, so swapping
Supertonic for another engine is a new file implementing `SynthesisEngine`
plus a one-line change at the call site. `engine_id` takes part in the clip
cache key, so two engines' outputs can coexist in the cache without colliding.
"""

from typing import Protocol

import numpy as np


class SynthesisEngine(Protocol):
    engine_id: str
    sample_rate: int

    def synthesize(self, text: str, language: str, voice: str) -> np.ndarray:
        """Return mono float32 samples in [-1.0, 1.0] at `self.sample_rate`."""
        ...
```

- [ ] **Step 4: Write `germanaudio/tts/supertonic.py`**

```python
"""Supertonic 3 adapter.

Supertonic returns whatever its ONNX graph produces; this module is the
single place that normalizes it to the pipeline's contract of mono float32
in [-1, 1]. The conversions below are defensive on purpose — the exact dtype
is recorded by the listening test in Task 1, and being wrong here is a silent
volume or pitch bug rather than a crash.
"""

import numpy as np

from germanaudio.config import SAMPLE_RATE

_INT16_SCALE = 32768.0


class SupertonicEngine:
    engine_id = "supertonic-3"

    def __init__(self, tts=None, sample_rate: int = SAMPLE_RATE):
        if tts is None:
            from supertonic import TTS

            tts = TTS(auto_download=True)
        self._tts = tts
        self.sample_rate = sample_rate
        self._styles: dict[str, object] = {}

    def _style(self, voice: str):
        if voice not in self._styles:
            self._styles[voice] = self._tts.get_voice_style(voice_name=voice)
        return self._styles[voice]

    def synthesize(self, text: str, language: str, voice: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("cannot synthesize empty text")

        wav, _duration = self._tts.synthesize(
            text, voice_style=self._style(voice), lang=language
        )
        return _to_mono_float32(wav)


def _to_mono_float32(wav) -> np.ndarray:
    samples = np.asarray(wav)

    if samples.ndim > 1:
        # Average channels rather than dropping one, so a hard-panned
        # channel does not silently halve the volume.
        samples = samples.mean(axis=-1)

    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float32) / _INT16_SCALE
    else:
        samples = samples.astype(np.float32)

    return np.ascontiguousarray(samples)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest tests/test_supertonic.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 6: Verify the adapter against the real model**

```bash
python -c "from germanaudio.tts.supertonic import SupertonicEngine; import numpy as np; e = SupertonicEngine(); s = e.synthesize('das Fenster', 'de', 'M1'); print(s.dtype, s.shape, s.min(), s.max())"
```

Expected: `float32`, a one-dimensional shape of roughly `sample_rate * 1.2` samples, and min/max within `[-1, 1]`. If the range is far below 1.0 the output will be quiet — report the values rather than adding a gain factor unilaterally.

- [ ] **Step 7: Commit**

```bash
git add germanaudio/tts/ tests/conftest.py tests/test_supertonic.py
git commit -m "feat: synthesis engine protocol and Supertonic adapter"
```

---

### Task 5: Clip specification and content-hash cache

**Files:**
- Create: `germanaudio/synth.py`
- Test: `tests/test_synth.py`

**Interfaces:**
- Consumes: `Entry`, `VoiceConfig`, `SynthesisEngine`, `germanaudio.audio.read_wav` / `write_wav`
- Produces:
  - `germanaudio.synth.ClipSpec` — frozen dataclass `text: str`, `language: str`, `voice: str`; method `key(engine_id: str) -> str`
  - `germanaudio.synth.entry_clips(entry: Entry, voices: VoiceConfig) -> tuple[ClipSpec, ClipSpec, ClipSpec]` — returns `(word_de, translation_en, phrase_de)`
  - `germanaudio.synth.required_clips(entries: list[Entry], voices: VoiceConfig) -> list[ClipSpec]` — deduplicated, order preserved
  - `germanaudio.synth.clip_path(cache_dir: Path, spec: ClipSpec, engine_id: str) -> Path`
  - `germanaudio.synth.SynthesisReport` — frozen dataclass `synthesized: int`, `reused: int`, `failures: list[tuple[ClipSpec, str]]`
  - `germanaudio.synth.synthesize_all(entries, engine, cache_dir, voices, progress=None) -> SynthesisReport`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_synth.py`:

```python
import numpy as np
import pytest

from germanaudio.config import VoiceConfig
from germanaudio.models import Entry
from germanaudio.synth import (
    ClipSpec,
    clip_path,
    entry_clips,
    required_clips,
    synthesize_all,
)

VOICES = VoiceConfig(de="M1", en="F1")


def entry(german="das Fenster", english="the window", phrase="Mach das Fenster zu.", line=2):
    return Entry(german, english, phrase, line)


def test_entry_produces_three_clips_in_order():
    word, translation, phrase = entry_clips(entry(), VOICES)

    assert (word.text, word.language, word.voice) == ("das Fenster", "de", "M1")
    assert (translation.text, translation.language, translation.voice) == (
        "the window",
        "en",
        "F1",
    )
    assert (phrase.text, phrase.language, phrase.voice) == (
        "Mach das Fenster zu.",
        "de",
        "M1",
    )


def test_key_is_stable_for_identical_input():
    spec = ClipSpec("das Fenster", "de", "M1")
    assert spec.key("supertonic-3") == ClipSpec("das Fenster", "de", "M1").key("supertonic-3")


@pytest.mark.parametrize(
    "other",
    [
        ClipSpec("der Schlüssel", "de", "M1"),
        ClipSpec("das Fenster", "en", "M1"),
        ClipSpec("das Fenster", "de", "F1"),
    ],
)
def test_key_differs_on_text_language_or_voice(other):
    assert ClipSpec("das Fenster", "de", "M1").key("supertonic-3") != other.key("supertonic-3")


def test_key_differs_across_engines():
    spec = ClipSpec("das Fenster", "de", "M1")
    assert spec.key("supertonic-3") != spec.key("piper")


def test_required_clips_deduplicates_repeated_text():
    entries = [
        entry(german="das Buch", english="the book", phrase="Ich lese das Buch."),
        entry(german="das Buch", english="the book", phrase="Das Buch ist gut."),
    ]

    specs = required_clips(entries, VOICES)

    # "das Buch" and "the book" each collapse to one clip; two distinct phrases remain.
    assert len(specs) == 4


def test_synthesize_all_writes_one_file_per_unique_clip(tmp_path, fake_engine):
    entries = [entry()]

    report = synthesize_all(entries, fake_engine, tmp_path, VOICES)

    assert report.synthesized == 3
    assert report.reused == 0
    assert report.failures == []
    assert len(list(tmp_path.glob("*.wav"))) == 3


def test_repeated_slots_do_not_synthesize_twice(tmp_path, fake_engine):
    """The doubled drill reuses clips — it must not cost synthesis time."""
    synthesize_all([entry()], fake_engine, tmp_path, VOICES)

    assert len(fake_engine.calls) == 3


def test_second_run_reuses_the_cache(tmp_path, fake_engine):
    entries = [entry()]
    synthesize_all(entries, fake_engine, tmp_path, VOICES)
    fake_engine.calls.clear()

    report = synthesize_all(entries, fake_engine, tmp_path, VOICES)

    assert report.synthesized == 0
    assert report.reused == 3
    assert fake_engine.calls == []


def test_resumes_after_partial_run(tmp_path, fake_engine):
    entries = [
        entry(german="das Buch", english="the book", phrase="Ich lese das Buch."),
        entry(german="der Tag", english="the day", phrase="Der Tag war schön."),
    ]
    synthesize_all(entries[:1], fake_engine, tmp_path, VOICES)
    fake_engine.calls.clear()

    report = synthesize_all(entries, fake_engine, tmp_path, VOICES)

    assert report.reused == 3
    assert report.synthesized == 3


def test_one_failure_does_not_stop_the_run(tmp_path):
    class FlakyEngine:
        engine_id = "flaky"
        sample_rate = 44100

        def synthesize(self, text, language, voice):
            if text == "the window":
                raise RuntimeError("model blew up")
            return np.array([0.1, 0.2], dtype=np.float32)

    report = synthesize_all([entry()], FlakyEngine(), tmp_path, VOICES)

    assert report.synthesized == 2
    assert len(report.failures) == 1
    assert "model blew up" in report.failures[0][1]


def test_failed_clip_leaves_no_partial_file(tmp_path):
    class AlwaysFails:
        engine_id = "broken"
        sample_rate = 44100

        def synthesize(self, text, language, voice):
            raise RuntimeError("nope")

    report = synthesize_all([entry()], AlwaysFails(), tmp_path, VOICES)

    assert len(report.failures) == 3
    assert list(tmp_path.glob("*.wav")) == []


def test_progress_callback_receives_each_clip(tmp_path, fake_engine):
    seen = []
    synthesize_all([entry()], fake_engine, tmp_path, VOICES, progress=lambda i, n, s: seen.append((i, n)))

    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_clip_path_lives_under_the_cache_directory(tmp_path):
    spec = ClipSpec("das Fenster", "de", "M1")
    path = clip_path(tmp_path, spec, "supertonic-3")

    assert path.parent == tmp_path
    assert path.suffix == ".wav"
    assert path.stem == spec.key("supertonic-3")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_synth.py -v
```

Expected: all fail with `ModuleNotFoundError: No module named 'germanaudio.synth'`.

- [ ] **Step 3: Write `germanaudio/synth.py`**

```python
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
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    specs = required_clips(entries, voices)
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_synth.py -v
```

Expected: all 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add germanaudio/synth.py tests/test_synth.py
git commit -m "feat: clip specs and resumable content-hash synthesis cache"
```

---

### Task 6: Timeline assembly and MP3 encoding

**Files:**
- Create: `germanaudio/assemble.py`
- Test: `tests/test_assemble.py`

**Interfaces:**
- Consumes: `Entry`, `ClipSpec`, `entry_clips`, `clip_path`, `PauseConfig`, `VoiceConfig`, `read_wav`, `silence`
- Produces:
  - `germanaudio.assemble.Silence` — frozen dataclass `seconds: float`
  - `germanaudio.assemble.TimelineItem = ClipSpec | Silence`
  - `germanaudio.assemble.build_timeline(entries, voices, pauses) -> list[TimelineItem]` — pure, no I/O
  - `germanaudio.assemble.render(timeline, cache_dir, engine_id, sample_rate) -> np.ndarray`
  - `germanaudio.assemble.encode_mp3(samples, sample_rate, path, bitrate) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_assemble.py`:

```python
import numpy as np
import pytest

from germanaudio.assemble import Silence, build_timeline, encode_mp3, render
from germanaudio.config import PauseConfig, VoiceConfig
from germanaudio.models import Entry
from germanaudio.synth import ClipSpec, synthesize_all

VOICES = VoiceConfig(de="M1", en="F1")
PAUSES = PauseConfig()


def entry(german="das Fenster", english="the window", phrase="Mach das Fenster zu.", line=2):
    return Entry(german, english, phrase, line)


def test_timeline_has_five_clips_and_five_pauses_per_entry():
    timeline = build_timeline([entry()], VOICES, PAUSES)

    assert len(timeline) == 10
    assert sum(1 for item in timeline if isinstance(item, ClipSpec)) == 5
    assert sum(1 for item in timeline if isinstance(item, Silence)) == 5


def test_timeline_follows_the_de_en_de_en_phrase_pattern():
    timeline = build_timeline([entry()], VOICES, PAUSES)
    clips = [item for item in timeline if isinstance(item, ClipSpec)]

    assert [clip.text for clip in clips] == [
        "das Fenster",
        "the window",
        "das Fenster",
        "the window",
        "Mach das Fenster zu.",
    ]
    assert [clip.language for clip in clips] == ["de", "en", "de", "en", "de"]


def test_repeated_slots_reference_the_identical_clip_spec():
    """Slots 3 and 4 must be the same specs as 1 and 2, or the cache is missed."""
    timeline = build_timeline([entry()], VOICES, PAUSES)
    clips = [item for item in timeline if isinstance(item, ClipSpec)]

    assert clips[0] == clips[2]
    assert clips[1] == clips[3]


def test_timeline_alternates_clip_then_silence():
    timeline = build_timeline([entry()], VOICES, PAUSES)

    for index, item in enumerate(timeline):
        expected = ClipSpec if index % 2 == 0 else Silence
        assert isinstance(item, expected)


def test_pause_durations_come_from_config():
    pauses = PauseConfig(
        after_word=3.0,
        after_translation=0.5,
        after_repeat_word=0.6,
        after_repeat_translation=0.7,
        after_phrase=4.0,
    )

    timeline = build_timeline([entry()], VOICES, pauses)
    gaps = [item.seconds for item in timeline if isinstance(item, Silence)]

    assert gaps == [3.0, 0.5, 0.6, 0.7, 4.0]


def test_entries_appear_in_file_order():
    entries = [
        entry(german="eins", english="one", phrase="Eins ist die erste Zahl."),
        entry(german="zwei", english="two", phrase="Zwei kommt nach eins."),
    ]

    timeline = build_timeline(entries, VOICES, PAUSES)
    clips = [item for item in timeline if isinstance(item, ClipSpec)]

    assert clips[0].text == "eins"
    assert clips[5].text == "zwei"


def test_empty_entry_list_produces_empty_timeline():
    assert build_timeline([], VOICES, PAUSES) == []


def test_render_concatenates_clips_and_silence(tmp_path, fake_engine):
    entries = [entry()]
    synthesize_all(entries, fake_engine, tmp_path, VOICES)
    timeline = build_timeline(entries, VOICES, PAUSES)

    samples = render(timeline, tmp_path, fake_engine.engine_id, fake_engine.sample_rate)

    assert samples.dtype == np.float32
    expected_silence = sum(item.seconds for item in timeline if isinstance(item, Silence))
    assert len(samples) > int(expected_silence * fake_engine.sample_rate)


def test_render_total_duration_equals_sum_of_parts(tmp_path, fake_engine):
    from germanaudio.audio import read_wav
    from germanaudio.synth import clip_path

    entries = [entry()]
    synthesize_all(entries, fake_engine, tmp_path, VOICES)
    timeline = build_timeline(entries, VOICES, PAUSES)

    expected = 0
    for item in timeline:
        if isinstance(item, Silence):
            expected += int(round(item.seconds * fake_engine.sample_rate))
        else:
            clip, _ = read_wav(clip_path(tmp_path, item, fake_engine.engine_id))
            expected += len(clip)

    samples = render(timeline, tmp_path, fake_engine.engine_id, fake_engine.sample_rate)

    assert len(samples) == expected


def test_render_reports_a_missing_clip_by_text(tmp_path, fake_engine):
    timeline = build_timeline([entry()], VOICES, PAUSES)

    with pytest.raises(FileNotFoundError, match="das Fenster"):
        render(timeline, tmp_path, fake_engine.engine_id, fake_engine.sample_rate)


def test_encode_mp3_writes_a_real_mp3(tmp_path):
    samples = (0.3 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)).astype(np.float32)
    path = tmp_path / "track.mp3"

    encode_mp3(samples, 44100, path, bitrate=64)

    assert path.exists()
    assert path.stat().st_size > 1000
    # MP3 frames begin with a sync word; ID3 tags begin with "ID3".
    head = path.read_bytes()[:3]
    assert head[:2] == b"\xff\xfb" or head == b"ID3" or head[0] == 0xFF


def test_encode_mp3_creates_the_output_directory(tmp_path):
    samples = np.zeros(44100, dtype=np.float32)
    path = tmp_path / "nested" / "dir" / "track.mp3"

    encode_mp3(samples, 44100, path, bitrate=64)

    assert path.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_assemble.py -v
```

Expected: all fail with `ModuleNotFoundError: No module named 'germanaudio.assemble'`.

- [ ] **Step 3: Write `germanaudio/assemble.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_assemble.py -v
```

Expected: all 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add germanaudio/assemble.py tests/test_assemble.py
git commit -m "feat: timeline assembly and MP3 encoding"
```

---

### Task 7: CLI and end-to-end run

**Files:**
- Create: `germanaudio/cli.py`
- Create: `germanaudio/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 2–6
- Produces: `germanaudio.cli.main(argv: list[str] | None = None) -> int` — process exit code, `0` on success

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
import numpy as np
import pytest

from germanaudio import cli

VOCAB = (
    "german\tenglish\tphrase\n"
    "das Fenster\tthe window\tBitte mach das Fenster zu, es ist kalt.\n"
    "der Schlüssel\tthe key\tIch habe meinen Schlüssel vergessen.\n"
)


@pytest.fixture
def vocab_file(tmp_path):
    path = tmp_path / "vocab.tsv"
    path.write_text(VOCAB, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def stub_engine(monkeypatch, fake_engine):
    """Never let the CLI tests touch the real model."""
    monkeypatch.setattr(cli, "build_engine", lambda: fake_engine)
    return fake_engine


def test_check_mode_validates_without_synthesizing(vocab_file, tmp_path, capsys, stub_engine):
    exit_code = cli.main([str(vocab_file), "--check"])

    assert exit_code == 0
    assert stub_engine.calls == []
    assert "2 entries" in capsys.readouterr().out


def test_check_mode_reports_bad_lines_and_fails(tmp_path, capsys):
    bad = tmp_path / "bad.tsv"
    bad.write_text("german\tenglish\tphrase\nonly\ttwo\n", encoding="utf-8")

    exit_code = cli.main([str(bad), "--check"])

    assert exit_code == 1
    assert "line 2" in capsys.readouterr().err


def test_full_run_produces_an_mp3(vocab_file, tmp_path):
    out = tmp_path / "track.mp3"

    exit_code = cli.main(
        [str(vocab_file), "-o", str(out), "--cache-dir", str(tmp_path / "cache")]
    )

    assert exit_code == 0
    assert out.exists()
    assert out.stat().st_size > 1000


def test_limit_flag_truncates_the_entry_list(vocab_file, tmp_path, stub_engine):
    cli.main(
        [
            str(vocab_file),
            "-o", str(tmp_path / "track.mp3"),
            "--cache-dir", str(tmp_path / "cache"),
            "--limit", "1",
        ]
    )

    assert len(stub_engine.calls) == 3


def test_warnings_are_printed_but_do_not_fail_the_run(tmp_path, capsys):
    path = tmp_path / "warn.tsv"
    path.write_text(
        "german\tenglish\tphrase\nder Stuhl\tthe chair\tIch sitze am Tisch.\n",
        encoding="utf-8",
    )

    exit_code = cli.main(
        [str(path), "-o", str(tmp_path / "t.mp3"), "--cache-dir", str(tmp_path / "c")]
    )

    assert exit_code == 0
    assert "der Stuhl" in capsys.readouterr().err


def test_synthesis_failure_aborts_before_writing_a_track(vocab_file, tmp_path, monkeypatch, capsys):
    class BrokenEngine:
        engine_id = "broken"
        sample_rate = 44100

        def synthesize(self, text, language, voice):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(cli, "build_engine", lambda: BrokenEngine())
    out = tmp_path / "track.mp3"

    exit_code = cli.main([str(vocab_file), "-o", str(out), "--cache-dir", str(tmp_path / "c")])

    assert exit_code == 1
    assert not out.exists()
    assert "model unavailable" in capsys.readouterr().err


def test_missing_input_file_fails_cleanly(tmp_path, capsys):
    exit_code = cli.main([str(tmp_path / "nope.tsv"), "--check"])

    assert exit_code == 1
    assert capsys.readouterr().err != ""
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_cli.py -v
```

Expected: all fail with `ImportError: cannot import name 'cli' from 'germanaudio'`.

- [ ] **Step 3: Write `germanaudio/cli.py`**

```python
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
```

- [ ] **Step 4: Write `germanaudio/__main__.py`**

```python
from germanaudio.cli import main

raise SystemExit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest tests/test_cli.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 6: Run the whole suite**

```bash
python -m pytest -v
```

Expected: 61 tests PASS, nothing skipped.

- [ ] **Step 7: End-to-end run against the real model**

Create a small `sample.tsv` by hand (eight entries is enough) and run:

```bash
python -m germanaudio sample.tsv --check
python -m germanaudio sample.tsv -o out/sample.mp3
```

Expected: validation reports 8 entries; synthesis reports 24 new clips; a roughly two-minute MP3 appears in `out/`. Listen to it end-to-end and check the pauses feel right before running the full 500.

- [ ] **Step 8: Commit**

```bash
git add germanaudio/cli.py germanaudio/__main__.py tests/test_cli.py
git commit -m "feat: CLI wiring the load, synthesize and assemble stages"
```

---

## Self-review notes

Checked against the spec:

- Input format, both delimiters, BOM, comments, blank lines — Task 2
- All-problems-at-once validation with line numbers — Task 2
- Duplicate and substring warnings as warnings, not errors — Task 2
- Three clips per entry, five placements, cache key over text/language/voice/engine — Tasks 5 and 6
- Resume after interruption — Task 5
- Failed clip aborts before assembly — Tasks 5 and 7
- Engine protocol with `engine_id` in the cache key — Task 4
- Pause config block, 44.1 kHz mono, 64 kbps MP3 via `lameenc` — Tasks 1, 3, 6
- Output-directory check before work starts — `encode_mp3` creates it; the `--check` path exits before any synthesis
- CLI surface `--check` and `--limit` — Task 7
- Listening test as the first gate — Task 1

One deliberate deviation from the spec: the spec describes checking that the output directory is writable before stage 2 begins. `encode_mp3` creates the directory tree instead. Creating it up front would mean an empty `out/` after a failed run, and a permission failure on a directory the process just created is vanishingly rare next to the failure modes that are actually covered.
