"""Read-aloud mode: plain German text in, one clip per sentence out.

Synthesizing sentence by sentence rather than the whole text at once keeps
every engine input short, and gives prose the same cache properties as
vocabulary: an interrupted run resumes, and editing one sentence
re-synthesizes one clip. The output is an ordinary timeline, so rendering
and encoding are shared with the vocabulary track unchanged.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from germanaudio.assemble import Silence, TimelineItem
from germanaudio.config import TextPauseConfig
from germanaudio.loader import LoadError
from germanaudio.synth import ClipSpec


# Space + em dash + space, the same separator the Hörplayer splits section titles on.
TITLE_SEPARATOR = " — "

# "{37|siebenunddreißig}": the read-along text shows the left side, the voice reads the right side.
_OVERRIDE = re.compile(r"\{([^{}|]+)\|([^{}|]+)\}")


class OverrideError(ValueError):
    """A malformed {shown|spoken} override; the message quotes the text."""


def _resolve(text: str, side: int) -> str:
    resolved = _OVERRIDE.sub(lambda m: m.group(side), text)
    leftover = _OVERRIDE.sub("", text)
    if "{" in leftover or "}" in leftover:
        raise OverrideError(f"malformed {{shown|spoken}} override in {text!r}")
    return resolved


def spoken_text(text: str) -> str:
    """The text as the voice reads it: every override replaced by its spoken side."""
    return _resolve(text, 2)


def shown_text(text: str) -> str:
    """The text as the read-along shows it: every override replaced by its shown side."""
    return _resolve(text, 1)


@dataclass(frozen=True)
class Heading:
    """A section heading line ("# Die Entscheidung — The decision").

    Only the German part is spoken; the full title becomes the chapter mark
    the Hörplayer shows (German big, English underneath).
    """

    german: str
    english: str | None

    @property
    def title(self) -> str:
        german = shown_text(self.german)
        return f"{german}{TITLE_SEPARATOR}{shown_text(self.english)}" if self.english else german

    @property
    def spoken(self) -> str:
        german = spoken_text(self.german)
        return german if german[-1] in ".!?…" else f"{german}."


Paragraphs = list[list[str] | Heading]
Marks = dict[int, str]  # timeline index -> chapter title

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

# Terminal punctuation, then any closing quotes or brackets, then whitespace.
_CANDIDATE_END = re.compile(r"[.!?…]+[\"'“”»«’)\]]*\s+")


def _is_sentence_end(paragraph: str, match: re.Match) -> bool:
    """Reject the two commonest false ends in German prose.

    A period followed by a lowercase word is almost always an abbreviation
    ("ca. zehn", "bzw. auch"), and a period straight after a digit is an
    ordinal ("am 3. Mai"). Getting one of these wrong only costs a short
    pause mid-sentence, so the rules stay deliberately small.
    """
    next_char = paragraph[match.end()]
    if next_char.islower():
        return False
    punctuation = match.group().rstrip()
    after_digit = match.start() > 0 and paragraph[match.start() - 1].isdigit()
    return not (punctuation == "." and after_digit)


def _split_sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    overrides = [m.span() for m in _OVERRIDE.finditer(paragraph)]
    for match in _CANDIDATE_END.finditer(paragraph):
        if any(first < match.start() < last for first, last in overrides):
            continue  # a period inside {z. B.|zum Beispiel} is not a sentence end
        if _is_sentence_end(paragraph, match):
            sentences.append(paragraph[start:match.end()].strip())
            start = match.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _parse_heading(line: str) -> Heading | None:
    text = " ".join(line.lstrip()[1:].split())
    if not text:
        return None
    german, _, english = text.partition(TITLE_SEPARATOR.strip())
    return Heading(german.strip(), english.strip() or None)


def split_text(text: str) -> Paragraphs:
    """Paragraphs at blank lines, sentences within them, line wraps joined.

    A line starting with "#" is a section heading and always stands alone,
    even without blank lines around it.
    """
    paragraphs: Paragraphs = []

    def flush(lines: list[str]) -> None:
        joined = " ".join(" ".join(lines).split())
        if joined:
            paragraphs.append(_split_sentences(joined))
        lines.clear()

    for block in _PARAGRAPH_BREAK.split(text):
        pending: list[str] = []
        for line in block.splitlines():
            if line.lstrip().startswith("#"):
                flush(pending)
                heading = _parse_heading(line)
                if heading is not None:
                    paragraphs.append(heading)
            else:
                pending.append(line)
        flush(pending)
    return paragraphs


def load_text(path: Path) -> Paragraphs:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise LoadError([f"cannot read {path}: {error}"]) from error
    except UnicodeDecodeError as error:
        raise LoadError([f"{path} is not valid UTF-8"]) from error

    problems = override_problems(text.splitlines())
    if problems:
        raise LoadError(problems)
    paragraphs = split_text(text)
    if not paragraphs:
        raise LoadError([f"{path} contains no text"])
    return paragraphs


def override_problems(lines: list[str], first_line_number: int = 1) -> list[str]:
    """"line N: …" for every line with a malformed {shown|spoken} override (an override must sit on one line)."""
    problems = []
    for number, line in enumerate(lines, start=first_line_number):
        try:
            spoken_text(line)
        except OverrideError as error:
            problems.append(f"line {number}: {error} (an override must be on one line)")
    return problems


def limit_sentences(paragraphs: Paragraphs, limit: int) -> Paragraphs:
    """The first `limit` sentences, keeping their paragraph grouping."""
    kept: Paragraphs = []
    remaining = limit
    for block in paragraphs:
        if remaining <= 0:
            break
        if isinstance(block, Heading):
            kept.append(block)
            remaining -= 1
            continue
        kept.append(block[:remaining])
        remaining -= len(kept[-1])
    return kept


def story_title(paragraphs: Paragraphs) -> str | None:
    """The story's first sentence without its final punctuation, e.g. "Ioanas neues Leben"."""
    for block in paragraphs:
        if isinstance(block, Heading):
            return None
        return shown_text(block[0]).rstrip(".!?… ")
    return None


def text_timeline(paragraphs: Paragraphs, voice: str, pauses: TextPauseConfig) -> list[TimelineItem]:
    timeline: list[TimelineItem] = []
    for block in paragraphs:
        if isinstance(block, Heading):
            timeline.append(ClipSpec(block.spoken, "de", voice))
            timeline.append(Silence(pauses.after_paragraph))
            continue
        sentences = block
        for index, sentence in enumerate(sentences):
            is_last = index == len(sentences) - 1
            timeline.append(ClipSpec(spoken_text(sentence), "de", voice))
            timeline.append(Silence(pauses.after_paragraph if is_last else pauses.after_sentence))
    return timeline


def text_lines(paragraphs: Paragraphs) -> dict[int, str]:
    """The read-along text of every spoken clip in `text_timeline`, keyed by timeline index.

    Sentences as written; the first sentence of a paragraph after the first starts with a line break, and a heading
    is "# " + its full title — the Hörplayer's SYLT conventions.
    """
    lines: dict[int, str] = {}
    index = 0
    for block in paragraphs:
        if isinstance(block, Heading):
            lines[index] = f"# {block.title}"
            index += 2
            continue
        for position, sentence in enumerate(block):
            new_paragraph = position == 0 and bool(lines)
            shown = shown_text(sentence)
            lines[index] = f"\n{shown}" if new_paragraph else shown
            index += 2
    return lines


def text_marks(paragraphs: Paragraphs) -> Marks:
    """Where each heading's spoken clip sits in `text_timeline`, keyed by timeline index."""
    marks: Marks = {}
    index = 0
    for block in paragraphs:
        if isinstance(block, Heading):
            marks[index] = block.title
            index += 2
        else:
            index += 2 * len(block)
    return marks
