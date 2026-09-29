import pytest

from germanaudio.assemble import Silence
from germanaudio.config import TextPauseConfig
from germanaudio.loader import LoadError
from germanaudio.reader import limit_sentences, load_text, split_text, text_timeline
from germanaudio.synth import ClipSpec

PAUSES = TextPauseConfig(after_sentence=0.5, after_paragraph=2.0)


def test_splits_sentences_on_terminal_punctuation():
    assert split_text("Hallo. Wie geht's? Gut!") == [["Hallo.", "Wie geht's?", "Gut!"]]


def test_blank_lines_separate_paragraphs():
    text = "Erster Absatz.\n\n\n  \nZweiter Absatz. Noch ein Satz."

    assert split_text(text) == [
        ["Erster Absatz."],
        ["Zweiter Absatz.", "Noch ein Satz."],
    ]


def test_line_wraps_inside_a_paragraph_are_joined():
    assert split_text("Ein Satz,\n  der weitergeht.") == [["Ein Satz, der weitergeht."]]


def test_closing_quote_stays_with_its_sentence():
    text = "Er sagte: „Komm her.“ Dann ging er."

    assert split_text(text) == [["Er sagte: „Komm her.“", "Dann ging er."]]


def test_ellipsis_ends_a_sentence():
    assert split_text("Na ja... Vielleicht morgen.") == [["Na ja...", "Vielleicht morgen."]]


def test_period_before_a_lowercase_word_does_not_split():
    # Abbreviations like "bzw." or "ca." are almost always followed by lowercase.
    assert split_text("Es kostet ca. zehn Euro.") == [["Es kostet ca. zehn Euro."]]


def test_ordinal_number_does_not_split():
    assert split_text("Wir fahren am 3. Mai los.") == [["Wir fahren am 3. Mai los."]]


def test_text_without_final_punctuation_keeps_its_last_sentence():
    assert split_text("Erster Satz. Ohne Punkt") == [["Erster Satz.", "Ohne Punkt"]]


def test_whitespace_only_text_has_no_paragraphs():
    assert split_text("  \n\n \t ") == []


def test_timeline_pauses_between_sentences_and_longer_after_paragraphs():
    timeline = text_timeline([["A.", "B."], ["C."]], "M1", PAUSES)

    assert timeline == [
        ClipSpec("A.", "de", "M1"),
        Silence(0.5),
        ClipSpec("B.", "de", "M1"),
        Silence(2.0),
        ClipSpec("C.", "de", "M1"),
        Silence(2.0),
    ]


def test_limit_keeps_the_first_n_sentences_across_paragraphs():
    paragraphs = [["A.", "B."], ["C.", "D."], ["E."]]

    assert limit_sentences(paragraphs, 3) == [["A.", "B."], ["C."]]


def test_limit_larger_than_the_text_keeps_everything():
    paragraphs = [["A."], ["B."]]

    assert limit_sentences(paragraphs, 10) == paragraphs


def test_load_text_reads_utf8_with_bom(tmp_path):
    path = tmp_path / "story.txt"
    path.write_text("Grüße aus München.", encoding="utf-8-sig")

    assert load_text(path) == [["Grüße aus München."]]


def test_load_text_rejects_an_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("\n  \n", encoding="utf-8")

    with pytest.raises(LoadError, match="no text"):
        load_text(path)


def test_load_text_rejects_non_utf8(tmp_path):
    path = tmp_path / "latin1.txt"
    path.write_bytes("Grüße".encode("latin-1"))

    with pytest.raises(LoadError, match="UTF-8"):
        load_text(path)


def test_load_text_reports_a_missing_file(tmp_path):
    with pytest.raises(LoadError, match="cannot read"):
        load_text(tmp_path / "missing.txt")


def test_leading_period_does_not_read_the_paragraphs_last_character():
    # paragraph[match.start() - 1] wraps to the final "5" when the match is at 0.
    assert split_text(". Seite 5") == [[".", "Seite 5"]]
