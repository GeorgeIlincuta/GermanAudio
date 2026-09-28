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
    except UnicodeDecodeError as error:
        raise LoadError([f"{path} is not valid UTF-8"]) from error

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
        # A line of only whitespace is a blank line, the same copy-paste
        # hazard the pipe fallback exists for, and should be skipped like
        # any other blank. But "\t\t" — a genuine three-empty-column row —
        # must still be reported as an error, so the check only fires when
        # the delimiter itself is absent.
        if (not raw.strip() and delimiter not in raw) or raw.lstrip().startswith("#"):
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
