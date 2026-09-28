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


def test_skips_whitespace_only_lines(tmp_path):
    """A blank line that picked up trailing whitespace on the way out of a
    chat window — the exact copy-paste hazard the pipe fallback exists for
    — must be skipped like any other blank line, not reported as a bad
    column count.
    """
    path = write(
        tmp_path,
        "german\tenglish\tphrase\n"
        "das Buch\tthe book\tIch lese jeden Abend ein gutes Buch.\n"
        "   \n"
        "der Stift\tthe pen\tMein Stift ist leer.\n",
    )

    result = load_vocab(path)

    assert len(result.entries) == 2
    assert result.entries[1].line_number == 4


def test_all_whitespace_columns_is_still_an_error(tmp_path):
    """A line of exactly three empty columns (tabs with nothing between
    them) is a genuine malformed row, not a blank line, and must still be
    reported — this guards against re-loosening the whitespace-only skip
    into swallowing this case too.
    """
    path = write(
        tmp_path,
        "german\tenglish\tphrase\n"
        "das Buch\tthe book\tIch lese jeden Abend ein gutes Buch.\n"
        "\t\t\n",
    )

    with pytest.raises(LoadError) as excinfo:
        load_vocab(path)

    assert any("line 3" in problem for problem in excinfo.value.problems)


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


def test_rejects_invalid_utf8(tmp_path):
    path = tmp_path / "invalid.tsv"
    path.write_bytes(b"german\tenglish\tphrase\n\xff\xfe invalid\tx\ty\n")

    with pytest.raises(LoadError) as excinfo:
        load_vocab(path)

    assert any("utf-8" in problem.lower() for problem in excinfo.value.problems)
