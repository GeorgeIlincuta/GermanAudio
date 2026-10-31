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


# --- section headings ("# Deutsch — English") -------------------------------------------------

from germanaudio.reader import Heading, story_title  # noqa: E402


def test_hash_line_becomes_a_heading_split_on_the_em_dash():
    assert split_text("# Die Entscheidung — The decision\nEs beginnt.") == [
        Heading("Die Entscheidung", "The decision"),
        ["Es beginnt."],
    ]


def test_heading_without_translation_keeps_german_only():
    heading = split_text("#Das Ende")[0]

    assert heading == Heading("Das Ende", None)
    assert heading.title == "Das Ende"


def test_heading_title_is_german_dash_english():
    assert Heading("Die Entscheidung", "The decision").title == "Die Entscheidung — The decision"


def test_only_the_german_part_is_spoken_and_ends_with_a_period():
    assert Heading("Die Entscheidung", "The decision").spoken == "Die Entscheidung."
    assert Heading("Wie viel ist drin?", "How much is in it?").spoken == "Wie viel ist drin?"


def test_heading_in_the_middle_of_a_block_splits_the_paragraphs_around_it():
    assert split_text("Vorher.\n# Mitte — Middle\nNachher.") == [
        ["Vorher."],
        Heading("Mitte", "Middle"),
        ["Nachher."],
    ]


def test_empty_heading_lines_are_ignored():
    assert split_text("#\n#   \nText.") == [["Text."]]


def test_heading_is_spoken_then_followed_by_the_paragraph_pause():
    timeline = text_timeline([Heading("Anfang", "Start"), ["A."]], "M1", PAUSES)

    assert timeline == [
        ClipSpec("Anfang.", "de", "M1"),
        Silence(2.0),
        ClipSpec("A.", "de", "M1"),
        Silence(2.0),
    ]


def test_limit_counts_a_heading_as_one_sentence():
    paragraphs = [Heading("Anfang", None), ["A.", "B."], ["C."]]

    assert limit_sentences(paragraphs, 2) == [Heading("Anfang", None), ["A."]]


def test_story_title_is_the_first_sentence_without_its_period():
    paragraphs = split_text("Ioanas neues Leben. Eine Geschichte.\n\n# Die Entscheidung — The decision\nText.")

    assert story_title(paragraphs) == "Ioanas neues Leben"


def test_story_title_is_none_when_the_text_starts_with_a_heading_only():
    assert story_title([Heading("Anfang", None)]) is None


# --- spoken overrides {shown|spoken} -----------------------------------------------------------

from germanaudio.reader import OverrideError, shown_text, spoken_text, text_lines, text_marks  # noqa: E402


def test_overrides_resolve_per_side():
    assert spoken_text("Es waren {37|siebenunddreißig} Menschen.") == "Es waren siebenunddreißig Menschen."
    assert shown_text("Es waren {37|siebenunddreißig} Menschen.") == "Es waren 37 Menschen."


def test_override_with_a_period_does_not_split_the_sentence():
    assert split_text("Das ist {z. B.|zum Beispiel} gut. Ja.") == [["Das ist {z. B.|zum Beispiel} gut.", "Ja."]]


def test_timeline_speaks_and_lines_show():
    paragraphs = split_text("# {3|Drei} Tage — Three days\n\nEs waren {37|siebenunddreißig}.")
    clips = [i.text for i in text_timeline(paragraphs, "M1", PAUSES) if isinstance(i, ClipSpec)]
    assert clips == ["Drei Tage.", "Es waren siebenunddreißig."]
    assert list(text_lines(paragraphs).values()) == ["# 3 Tage — Three days", "\nEs waren 37."]
    assert text_marks(paragraphs) == {0: "3 Tage — Three days"}


def test_text_without_braces_is_unchanged():
    p = split_text("Der Titel.\n\n# Teil — Part\n\nEin Satz. Noch einer!\n")
    assert [i.text for i in text_timeline(p, "M1", PAUSES) if isinstance(i, ClipSpec)] == \
        ["Der Titel.", "Teil.", "Ein Satz.", "Noch einer!"]
    assert list(text_lines(p).values()) == ["Der Titel.", "# Teil — Part", "\nEin Satz.", "Noch einer!"]


@pytest.mark.parametrize("bad", ["{37 sieben", "{|x}", "{x|}", "a}b", "{a|b|c}"])
def test_bad_overrides_raise(bad):
    with pytest.raises(OverrideError):
        spoken_text(bad)
