"""Read-aloud mode: plain German text in, one clip per sentence out.

Synthesizing sentence by sentence rather than the whole text at once keeps
every engine input short, and gives prose the same cache properties as
vocabulary: an interrupted run resumes, and editing one sentence
re-synthesizes one clip. The output is an ordinary timeline, so rendering
and encoding are shared with the vocabulary track unchanged.
"""

import re
from pathlib import Path

from germanaudio.assemble import Silence, TimelineItem
from germanaudio.config import TextPauseConfig
from germanaudio.loader import LoadError
from germanaudio.synth import ClipSpec

Paragraphs = list[list[str]]

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
    for match in _CANDIDATE_END.finditer(paragraph):
        if _is_sentence_end(paragraph, match):
            sentences.append(paragraph[start:match.end()].strip())
            start = match.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_text(text: str) -> Paragraphs:
    """Paragraphs at blank lines, sentences within them, line wraps joined."""
    paragraphs: Paragraphs = []
    for block in _PARAGRAPH_BREAK.split(text):
        joined = " ".join(block.split())
        if joined:
            paragraphs.append(_split_sentences(joined))
    return paragraphs


def load_text(path: Path) -> Paragraphs:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise LoadError([f"cannot read {path}: {error}"]) from error
    except UnicodeDecodeError as error:
        raise LoadError([f"{path} is not valid UTF-8"]) from error

    paragraphs = split_text(text)
    if not paragraphs:
        raise LoadError([f"{path} contains no text"])
    return paragraphs


def limit_sentences(paragraphs: Paragraphs, limit: int) -> Paragraphs:
    """The first `limit` sentences, keeping their paragraph grouping."""
    kept: Paragraphs = []
    remaining = limit
    for sentences in paragraphs:
        if remaining <= 0:
            break
        kept.append(sentences[:remaining])
        remaining -= len(kept[-1])
    return kept


def text_timeline(paragraphs: Paragraphs, voice: str, pauses: TextPauseConfig) -> list[TimelineItem]:
    timeline: list[TimelineItem] = []
    for sentences in paragraphs:
        for index, sentence in enumerate(sentences):
            is_last = index == len(sentences) - 1
            timeline.append(ClipSpec(sentence, "de", voice))
            timeline.append(Silence(pauses.after_paragraph if is_last else pauses.after_sentence))
    return timeline
