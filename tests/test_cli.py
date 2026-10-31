import wave

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


def test_assembly_failure_reports_cache_path_and_fails_cleanly(
    vocab_file, tmp_path, monkeypatch, capsys
):
    """A corrupt cache entry (or a render/encode crash of any kind) must not
    crash with a traceback, must not leave a partial output file, and must
    tell the user where to look — the cache directory is not otherwise
    discoverable from the error alone.
    """
    cache_dir = tmp_path / "cache"

    def broken_render(*args, **kwargs):
        raise wave.Error("file does not start with RIFF id")

    monkeypatch.setattr(cli, "render", broken_render)
    out = tmp_path / "track.mp3"

    exit_code = cli.main(
        [str(vocab_file), "-o", str(out), "--cache-dir", str(cache_dir)]
    )

    assert exit_code == 1
    assert not out.exists()
    assert str(cache_dir) in capsys.readouterr().err


def test_missing_input_file_fails_cleanly(tmp_path, capsys):
    exit_code = cli.main([str(tmp_path / "nope.tsv"), "--check"])

    assert exit_code == 1
    assert capsys.readouterr().err != ""


STORY = "Es war einmal ein Fuchs. Er war sehr hungrig.\n\nAm Abend fand er Brot.\n"


@pytest.fixture
def story_file(tmp_path):
    path = tmp_path / "story.txt"
    path.write_text(STORY, encoding="utf-8")
    return path


def test_text_mode_reads_each_sentence_in_german(story_file, tmp_path, stub_engine):
    out = tmp_path / "story.mp3"

    exit_code = cli.main(
        [str(story_file), "--text", "-o", str(out), "--cache-dir", str(tmp_path / "c")]
    )

    assert exit_code == 0
    assert out.stat().st_size > 1000
    assert [(text, lang) for text, lang, _ in stub_engine.calls] == [
        ("Es war einmal ein Fuchs.", "de"),
        ("Er war sehr hungrig.", "de"),
        ("Am Abend fand er Brot.", "de"),
    ]


def test_text_check_mode_counts_without_synthesizing(story_file, capsys, stub_engine):
    exit_code = cli.main([str(story_file), "--text", "--check"])

    assert exit_code == 0
    assert stub_engine.calls == []
    assert "3 sentences in 2 paragraphs" in capsys.readouterr().out


def test_text_limit_reads_only_the_first_sentences(story_file, tmp_path, stub_engine):
    cli.main(
        [
            str(story_file), "--text", "--limit", "2",
            "-o", str(tmp_path / "s.mp3"), "--cache-dir", str(tmp_path / "c"),
        ]
    )

    assert len(stub_engine.calls) == 2


def test_text_mode_rejects_an_empty_file(tmp_path, capsys, stub_engine):
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")

    exit_code = cli.main([str(path), "--text", "--check"])

    assert exit_code == 1
    assert "no text" in capsys.readouterr().err


# --- title and chapter tags ------------------------------------------------------------------

from mutagen.id3 import ID3  # noqa: E402


def chapter_titles(path):
    return [c.sub_frames["TIT2"].text[0] for c in sorted(ID3(path).getall("CHAP"), key=lambda c: c.start_time)]


def test_vocab_track_gets_a_chapter_per_entry_and_the_file_name_as_title(vocab_file, tmp_path):
    out = tmp_path / "track.mp3"

    assert cli.main([str(vocab_file), "-o", str(out), "--cache-dir", str(tmp_path / "c")]) == 0

    assert ID3(out)["TIT2"].text == ["vocab"]
    assert chapter_titles(out) == ["das Fenster — the window", "der Schlüssel — the key"]


def test_title_flag_overrides_the_default_title(vocab_file, tmp_path):
    out = tmp_path / "track.mp3"

    cli.main([str(vocab_file), "-o", str(out), "--cache-dir", str(tmp_path / "c"), "--title", "B1 Vokabeln"])

    assert ID3(out)["TIT2"].text == ["B1 Vokabeln"]


HEADED_STORY = (
    "Der kleine Fuchs. Eine Geschichte.\n\n"
    "# Der Hunger — The hunger\nEr war sehr hungrig.\n\n"
    "# Das Brot — The bread\nAm Abend fand er Brot.\n"
)


def test_story_headings_become_spoken_chapters_titled_from_the_first_sentence(tmp_path, stub_engine):
    story = tmp_path / "story.txt"
    story.write_text(HEADED_STORY, encoding="utf-8")
    out = tmp_path / "story.mp3"

    assert cli.main([str(story), "--text", "-o", str(out), "--cache-dir", str(tmp_path / "c")]) == 0

    assert ID3(out)["TIT2"].text == ["Der kleine Fuchs"]
    assert chapter_titles(out) == ["Der Hunger — The hunger", "Das Brot — The bread"]
    spoken = [text for text, _, _ in stub_engine.calls]
    assert "Der Hunger." in spoken and "The hunger" not in " ".join(spoken)


def test_story_without_headings_gets_a_title_but_no_chapters(story_file, tmp_path):
    out = tmp_path / "story.mp3"

    cli.main([str(story_file), "--text", "-o", str(out), "--cache-dir", str(tmp_path / "c")])

    assert ID3(out)["TIT2"].text == ["Es war einmal ein Fuchs"]
    assert ID3(out).getall("CHAP") == []


def test_check_mode_reports_headings(tmp_path, capsys):
    story = tmp_path / "story.txt"
    story.write_text(HEADED_STORY, encoding="utf-8")

    assert cli.main([str(story), "--text", "--check"]) == 0

    assert "2 headings" in capsys.readouterr().out


def test_story_build_writes_read_along_text_and_vocab_does_not(tmp_path, stub_engine):
    from mutagen.id3 import ID3

    story = tmp_path / "s.txt"
    story.write_text("Der Titel.\n\n# Teil — Part\n\nEin Satz.\n", encoding="utf-8")
    out = tmp_path / "s.mp3"
    assert cli.main([str(story), "--text", "-o", str(out), "--cache-dir", str(tmp_path / "c")]) == 0
    texts = [t for t, _ in ID3(out).getall("SYLT")[0].text]
    assert texts == ["Der Titel.", "# Teil — Part", "\nEin Satz."]

    vocab = tmp_path / "v.tsv"
    vocab.write_text("german\tenglish\tphrase\ndas Haus\tthe house\tDas Haus ist groß.\n", encoding="utf-8")
    vout = tmp_path / "v.mp3"
    assert cli.main([str(vocab), "-o", str(vout), "--cache-dir", str(tmp_path / "c")]) == 0
    assert ID3(vout).getall("SYLT") == []


def test_story_build_is_a_geschichte_and_vocab_a_wortschatz(tmp_path, stub_engine):
    from mutagen.id3 import ID3

    story = tmp_path / "s.txt"
    story.write_text("Der Titel.\n\n# Teil — Part\n\nEin Satz.\n", encoding="utf-8")
    out = tmp_path / "s.mp3"
    assert cli.main([str(story), "--text", "-o", str(out), "--cache-dir", str(tmp_path / "c")]) == 0
    assert ID3(out)["TCON"].text == ["Geschichte"]

    vocab = tmp_path / "v.tsv"
    vocab.write_text("german\tenglish\tphrase\ndas Haus\tthe house\tDas Haus ist groß.\n", encoding="utf-8")
    vout = tmp_path / "v.mp3"
    assert cli.main([str(vocab), "-o", str(vout), "--cache-dir", str(tmp_path / "c")]) == 0
    assert ID3(vout)["TCON"].text == ["Wortschatz"]


# --- Artikel ---------------------------------------------------------------------------------

ARTIKEL = """---
title: Wie sich das Gedächtnis im Alter verändert
author: Elena Bernard
source: https://wissenschaft.de/artikel/x
saved: 2026-10-31
---

## Wortschatz

das Gedächtnis | the memory | Mein Gedächtnis für Namen ist nicht gut.
abrufen | to retrieve | Unter Stress kann man Wissen schlechter abrufen.

## Artikel

# {3|Drei} Tage — Three days

Es waren {37|siebenunddreißig}.
"""


@pytest.fixture
def artikel_file(tmp_path):
    path = tmp_path / "a.md"
    path.write_text(ARTIKEL, encoding="utf-8")
    return path


def test_article_build_is_words_then_article(artikel_file, tmp_path, stub_engine):
    out = tmp_path / "a.mp3"

    assert cli.main([str(artikel_file), "--article", "-o", str(out), "--cache-dir", str(tmp_path / "c")]) == 0

    t = ID3(out)
    assert t["TCON"].text == ["Artikel"]
    assert t["TIT2"].text == ["Wie sich das Gedächtnis im Alter verändert"]
    assert chapter_titles(out) == ["das Gedächtnis — the memory", "abrufen — to retrieve", "3 Tage — Three days"]
    article_chap = sorted(t.getall("CHAP"), key=lambda c: c.start_time)[2]
    assert t["TXXX:Artikel-Start"].text == [str(article_chap.start_time)]
    assert t["TPE1"].text == ["Elena Bernard"] and t["WOAS"].url == "https://wissenschaft.de/artikel/x"
    assert [x for x, _ in t.getall("SYLT")[0].text] == ["# 3 Tage — Three days", "\nEs waren 37."]
    spoken = [x for x, _, _ in stub_engine.calls]
    assert "Es waren siebenunddreißig." in spoken and "Drei Tage." in spoken


def test_article_parts_are_three_seconds_apart(artikel_file):
    from germanaudio.assemble import Silence

    plan = cli._article_timeline(cli._parse_args([str(artikel_file), "--article"]), cli.VoiceConfig())

    assert plan.timeline[20] == Silence(3.0)  # after 2 entries x 10 items
    assert plan.marks[21] == "3 Tage — Three days"
    assert plan.article == (21, {"author": "Elena Bernard", "source": "https://wissenschaft.de/artikel/x"})
    assert plan.genre == "Artikel"


def test_article_check_reports_both_parts(artikel_file, capsys, stub_engine):
    assert cli.main([str(artikel_file), "--article", "--check"]) == 0
    assert stub_engine.calls == []
    assert "loaded 2 entries and 1 sentences in 1 paragraphs and 1 headings" in capsys.readouterr().out


def test_article_without_vocab_cannot_be_built(tmp_path, capsys):
    path = tmp_path / "a.md"
    path.write_text("\n".join(l for l in ARTIKEL.splitlines() if " | " not in l) + "\n", encoding="utf-8")
    assert cli.main([str(path), "--article", "--check"]) == 0
    assert cli.main([str(path), "--article", "-o", str(tmp_path / "a.mp3"), "--cache-dir", str(tmp_path / "c")]) == 1
    assert "no vocabulary" in capsys.readouterr().err


def test_article_retag_round_trip(artikel_file, tmp_path, monkeypatch, fake_engine):
    from germanaudio.retag import audio_digest

    monkeypatch.setattr(cli, "engine_identity", lambda: (fake_engine.engine_id, fake_engine.sample_rate))
    out = tmp_path / "a.mp3"
    args = [str(artikel_file), "--article", "-o", str(out), "--cache-dir", str(tmp_path / "c")]
    assert cli.main(args) == 0
    before = ID3(out)
    snapshot = ([(c.start_time, c.sub_frames["TIT2"].text) for c in before.getall("CHAP")],
                before.getall("SYLT")[0].text, before["TXXX:Artikel-Start"].text, before["TCON"].text)
    digest = audio_digest(out)

    assert cli.main(args + ["--retag"]) == 0

    after = ID3(out)
    assert ([(c.start_time, c.sub_frames["TIT2"].text) for c in after.getall("CHAP")],
            after.getall("SYLT")[0].text, after["TXXX:Artikel-Start"].text, after["TCON"].text) == snapshot
    assert audio_digest(out) == digest


def test_article_and_text_are_exclusive(artikel_file, capsys):
    with pytest.raises(SystemExit):
        cli.main([str(artikel_file), "--article", "--text", "--check"])
