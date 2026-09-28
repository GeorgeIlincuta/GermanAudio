"""Shared test doubles."""

import hashlib

import numpy as np
import pytest


class FakeEngine:
    """Deterministic stand-in for a real TTS engine.

    Returns a distinct, reproducible tone per input so that tests can assert
    which clip landed where without listening to anything. Length scales with
    text length, which makes duration assertions meaningful.
    """

    engine_id = "fake"
    sample_rate = 44100

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def synthesize(self, text: str, language: str, voice: str) -> np.ndarray:
        self.calls.append((text, language, voice))

        digest = hashlib.sha1(text.encode("utf-8")).digest()
        frequency = 200 + digest[0]  # 200-455 Hz, stable per text
        duration = 0.05 * len(text)

        t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()
