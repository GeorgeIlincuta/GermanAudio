"""Core data types shared across the pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    """One vocabulary item: a word, its translation, and an example.

    `line_number` is 1-based and refers to the source file, so that any
    problem found downstream can be pointed back at a line the user can edit.
    """

    german: str
    english: str
    phrase: str
    line_number: int
