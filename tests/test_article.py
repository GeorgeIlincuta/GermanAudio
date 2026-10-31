import pytest

from germanaudio.article import load_article
from germanaudio.loader import LoadError
from germanaudio.reader import Heading

ARTICLE = """---
title: Wie sich das Gedächtnis im Alter verändert
author: Elena Bernard
published: 2026-10-23
category: Gesundheit & Medizin
source: https://wissenschaft.de/artikel/wie-sich-das-gedaechtnis-im-alter-veraendert
study: Greta Melega et al., Psychology and Aging
saved: 2026-10-31
---

## Wortschatz

das Gedächtnis | the memory | Mein Gedächtnis für Namen ist leider nicht besonders gut.
abrufen | to retrieve | Unter Stress kann man gelerntes Wissen oft schlechter abrufen.

## Artikel

# Wie sich das Gedächtnis im Alter verändert — How memory changes with age

Wie detailliert wir uns erinnern, verändert sich mit dem Alter.

# Autobiografische Erinnerungen — Autobiographical memories

Die Forschenden baten {37|siebenunddreißig} junge Menschen um Hilfe.
"""


def write(tmp_path, text):
    path = tmp_path / "a.md"
    path.write_text(text, encoding="utf-8")
    return path


def problems(tmp_path, text):
    with pytest.raises(LoadError) as error:
        load_article(write(tmp_path, text))
    return error.value.problems


def test_loads_meta_vocab_and_article(tmp_path):
    a = load_article(write(tmp_path, ARTICLE))

    assert a.meta["title"] == "Wie sich das Gedächtnis im Alter verändert"
    assert a.meta["author"] == "Elena Bernard"
    assert a.meta["published"] == "2026-10-23"
    assert len(a.meta) == 7
    assert [e.german for e in a.entries] == ["das Gedächtnis", "abrufen"]
    assert a.entries[0].line_number == 13
    assert isinstance(a.paragraphs[0], Heading)
    assert a.paragraphs[0].title == "Wie sich das Gedächtnis im Alter verändert — How memory changes with age"
    assert a.paragraphs[-1] == ["Die Forschenden baten {37|siebenunddreißig} junge Menschen um Hilfe."]
    assert a.warnings == []


def test_missing_title_is_an_error(tmp_path):
    assert any("title" in p for p in problems(tmp_path, ARTICLE.replace("title: Wie sich", "titel: Wie sich")))


def test_missing_article_section_is_an_error(tmp_path):
    assert any("## Artikel" in p for p in problems(tmp_path, ARTICLE.replace("## Artikel", "Artikel")))


def test_unknown_section_is_an_error(tmp_path):
    assert any("## Quellen" in p for p in problems(tmp_path, ARTICLE + "\n## Quellen\n\nEtwas.\n"))


def test_article_must_start_with_a_heading(tmp_path):
    text = ARTICLE.replace("# Wie sich das Gedächtnis im Alter verändert — How memory changes with age\n\n", "")
    assert any("first line under ## Artikel" in p for p in problems(tmp_path, text))


def test_bad_override_names_its_line(tmp_path):
    text = ARTICLE.replace("{37|siebenunddreißig}", "{37 siebenunddreißig")
    assert any(p.startswith("line 24:") for p in problems(tmp_path, text))


def test_empty_vocabulary_loads(tmp_path):
    text = ARTICLE.replace("das Gedächtnis | the memory | Mein Gedächtnis für Namen ist leider nicht besonders gut.\n", "")
    text = text.replace("abrufen | to retrieve | Unter Stress kann man gelerntes Wissen oft schlechter abrufen.\n", "")
    assert load_article(write(tmp_path, text)).entries == []


def test_warnings(tmp_path):
    text = (ARTICLE.replace("— Autobiographical memories", "")
            .replace("Wie detailliert", "Mit 37 Jahren")
            .replace("saved: 2026-10-31", "saved: 2026-10-31\nfoo: bar"))
    warnings = load_article(write(tmp_path, text)).warnings
    assert "line 23: heading has no English half" in warnings
    assert "line 21: digits outside an override" in warnings
    assert "line 9: unknown front-matter key 'foo'" in warnings
