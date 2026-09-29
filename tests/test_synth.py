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
    synthesize_clips,
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


def test_synthesize_clips_works_on_bare_specs_without_entries(tmp_path, fake_engine):
    specs = [
        ClipSpec("Erster Satz.", "de", "M1"),
        ClipSpec("Zweiter Satz.", "de", "M1"),
        ClipSpec("Erster Satz.", "de", "M1"),
    ]

    report = synthesize_clips(specs, fake_engine, tmp_path)

    assert report.synthesized == 2
    assert [call[0] for call in fake_engine.calls] == ["Erster Satz.", "Zweiter Satz."]
    for spec in specs:
        assert clip_path(tmp_path, spec, fake_engine.engine_id).exists()
