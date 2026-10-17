from pathlib import Path

import pytest
from mutagen.id3 import ID3

from germanaudio import cli
from germanaudio.retag import audio_digest

STORY = "Der Titel.\n\n# Teil — Part\n\nEin Satz. Noch einer.\n"


@pytest.fixture
def built(tmp_path, monkeypatch, fake_engine):
    """A story built with the fake engine (so its clips are cached), then stripped of its SYLT."""
    monkeypatch.setattr(cli, "build_engine", lambda: fake_engine)
    monkeypatch.setattr(cli, "engine_identity", lambda: (fake_engine.engine_id, fake_engine.sample_rate))
    story = tmp_path / "s.txt"
    story.write_text(STORY, encoding="utf-8")
    out = tmp_path / "s.mp3"
    cache = tmp_path / "c"
    assert cli.main([str(story), "--text", "-o", str(out), "--cache-dir", str(cache)]) == 0
    tags = ID3(out)
    tags.delall("SYLT")
    tags.save(out, v2_version=3)
    return story, out, cache


def retag(story: Path, out: Path, cache: Path, *extra: str) -> int:
    return cli.main([str(story), "--text", "--retag", "-o", str(out), "--cache-dir", str(cache), *extra])


def test_retag_adds_the_text_and_keeps_the_audio(built):
    story, out, cache = built
    before = audio_digest(out)
    chapters_before = [c.start_time for c in ID3(out).getall("CHAP")]

    assert retag(story, out, cache) == 0

    tags = ID3(out)
    assert audio_digest(out) == before
    assert [t for t, _ in tags.getall("SYLT")[0].text] == ["Der Titel.", "# Teil — Part", "\nEin Satz.", "Noch einer."]
    assert [c.start_time for c in tags.getall("CHAP")] == chapters_before


def test_retag_needs_no_engine(built, monkeypatch):
    story, out, cache = built
    monkeypatch.setattr(cli, "build_engine", lambda: pytest.fail("retag must not load the TTS model"))
    assert retag(story, out, cache) == 0


def test_retag_stops_on_a_missing_clip_and_changes_nothing(built, capsys):
    story, out, cache = built
    for wav in cache.rglob("*.wav"):
        wav.unlink()
        break
    original = out.read_bytes()

    assert retag(story, out, cache) == 1
    assert out.read_bytes() == original
    assert "not in the cache" in capsys.readouterr().err


def test_retag_refuses_when_the_text_no_longer_matches_the_audio(built, capsys, fake_engine):
    story, out, cache = built
    story.write_text(STORY.replace("Noch einer.", "Noch einer, der jetzt viel länger ist als vorher."), encoding="utf-8")
    cli.synthesize_clips(  # the new sentence is cached, but the MP3 still holds the old audio
        [s for s in cli.text_timeline(cli.load_text(story), cli.VoiceConfig().de, cli.TextPauseConfig())
         if isinstance(s, cli.ClipSpec)],
        fake_engine, cache,
    )
    original = out.read_bytes()

    assert retag(story, out, cache) == 1
    assert out.read_bytes() == original
    assert "does not match" in capsys.readouterr().err


def test_retag_of_a_missing_file_fails(tmp_path, monkeypatch, fake_engine):
    monkeypatch.setattr(cli, "engine_identity", lambda: (fake_engine.engine_id, fake_engine.sample_rate))
    story = tmp_path / "s.txt"
    story.write_text(STORY, encoding="utf-8")
    assert retag(story, tmp_path / "nope.mp3", tmp_path / "c") == 1
