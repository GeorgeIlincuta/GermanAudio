"""Artikel mode input: one file with front matter, a vocabulary section and the article text.

    ---
    title: Wie sich das Gedächtnis im Alter verändert
    author: Elena Bernard
    ---

    ## Wortschatz

    das Gedächtnis | the memory | Mein Gedächtnis für Namen ist leider nicht besonders gut.

    ## Artikel

    # Wie sich das Gedächtnis im Alter verändert — How memory changes with age

    Wie detailliert wir uns an unsere Erlebnisse erinnern, …

The front matter is never spoken; it becomes tag frames. The vocabulary follows the TSV rules (no header row), the
article the --text rules plus {shown|spoken} overrides.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from germanaudio.loader import LoadError, detect_delimiter, parse_vocab_rows
from germanaudio.models import Entry
from germanaudio.reader import Heading, OverrideError, Paragraphs, split_text, spoken_text

META_KEYS = ("title", "author", "published", "category", "source", "study", "saved")

_VOCAB_SECTION = "## Wortschatz"
_ARTICLE_SECTION = "## Artikel"
_DIGIT = re.compile(r"\d")
_OVERRIDE_SPAN = re.compile(r"\{[^{}|]+\|[^{}|]+\}")


@dataclass(frozen=True)
class Article:
    meta: dict[str, str]  # only the keys present in the file
    entries: list[Entry]  # may be empty (fine for --check, refused by a build)
    paragraphs: Paragraphs  # the first block is a Heading
    warnings: list[str]


def _front_matter(lines: list[str], problems: list[str], warnings: list[str]) -> tuple[dict[str, str], int]:
    """The key/value pairs and the index of the first line after the closing ---."""
    if not lines or lines[0].strip() != "---":
        problems.append("line 1: the file must start with --- (front matter)")
        return {}, 0
    meta: dict[str, str] = {}
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() == "---":
            return meta, index + 1
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if not separator or not value:
            problems.append(f"line {index + 1}: front matter must be 'key: value', found {line!r}")
        elif key not in META_KEYS:
            warnings.append(f"line {index + 1}: unknown front-matter key {key!r}")
        else:
            meta[key] = value
    problems.append("line 1: front matter is not closed with ---")
    return meta, len(lines)


def _check_article_lines(lines: list[str], first: int, problems: list[str], warnings: list[str]) -> None:
    """Per-line checks that need the file line number: overrides, digits, headings without English."""
    for index in range(first, len(lines)):
        line = lines[index]
        number = index + 1
        try:
            spoken_text(line)
        except OverrideError as error:
            problems.append(f"line {number}: {error}")
            continue
        if _DIGIT.search(_OVERRIDE_SPAN.sub("", line)):
            warnings.append(f"line {number}: digits outside an override")
        stripped = line.lstrip()
        if stripped.startswith("#") and "—" not in stripped:
            warnings.append(f"line {number}: heading has no English half")


def load_article(path: Path) -> Article:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise LoadError([f"cannot read {path}: {error}"]) from error
    except UnicodeDecodeError as error:
        raise LoadError([f"{path} is not valid UTF-8"]) from error

    lines = text.splitlines()
    problems: list[str] = []
    warnings: list[str] = []
    meta, body_start = _front_matter(lines, problems, warnings)
    if "title" not in meta:
        problems.append("front matter has no title")

    vocab_at = article_at = None
    for index in range(body_start, len(lines)):
        stripped = lines[index].strip()
        if not stripped.startswith("## "):
            continue
        if stripped == _VOCAB_SECTION and vocab_at is None and article_at is None:
            vocab_at = index
        elif stripped == _ARTICLE_SECTION and vocab_at is not None and article_at is None:
            article_at = index
        else:
            problems.append(f"line {index + 1}: unexpected section {stripped!r} "
                            f"(expected {_VOCAB_SECTION}, then {_ARTICLE_SECTION})")
    if vocab_at is None:
        problems.append(f"no {_VOCAB_SECTION} section")
    if article_at is None:
        problems.append(f"no {_ARTICLE_SECTION} section after {_VOCAB_SECTION}")
    if problems:
        raise LoadError(problems)

    rows = [(index + 1, lines[index]) for index in range(vocab_at + 1, article_at)
            if lines[index].strip() and not lines[index].lstrip().startswith("#")]
    entries: list[Entry] = []
    if rows:
        entries, row_problems, row_warnings = parse_vocab_rows(rows, detect_delimiter(rows[0][1]))
        problems += row_problems
        warnings += row_warnings

    _check_article_lines(lines, article_at + 1, problems, warnings)
    if problems:
        raise LoadError(problems)

    paragraphs = split_text("\n".join(lines[article_at + 1:]))
    if not paragraphs:
        raise LoadError([f"line {article_at + 1}: {_ARTICLE_SECTION} has no text"])
    if not isinstance(paragraphs[0], Heading):
        raise LoadError([f"line {article_at + 1}: the first line under {_ARTICLE_SECTION} must be a # heading"])
    return Article(meta, entries, paragraphs, warnings)
