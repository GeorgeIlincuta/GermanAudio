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
