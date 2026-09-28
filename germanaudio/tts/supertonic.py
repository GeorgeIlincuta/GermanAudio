"""Supertonic 3 adapter.

Supertonic returns whatever its ONNX graph produces; this module is the
single place that normalizes it to the pipeline's contract of mono float32
in [-1, 1]. The conversions below are defensive on purpose — the exact dtype
is recorded by the listening test in Task 1, and being wrong here is a silent
volume or pitch bug rather than a crash.
"""

import numpy as np

from germanaudio.config import SAMPLE_RATE

_INT16_SCALE = 32768.0


class SupertonicEngine:
    engine_id = "supertonic-3"

    def __init__(self, tts=None, sample_rate: int = SAMPLE_RATE):
        if tts is None:
            from supertonic import TTS

            tts = TTS(auto_download=True)
        self._tts = tts
        self.sample_rate = sample_rate
        self._styles: dict[str, object] = {}

    def _style(self, voice: str):
        if voice not in self._styles:
            self._styles[voice] = self._tts.get_voice_style(voice_name=voice)
        return self._styles[voice]

    def synthesize(self, text: str, language: str, voice: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("cannot synthesize empty text")

        wav, _duration = self._tts.synthesize(
            text, voice_style=self._style(voice), lang=language
        )
        return _to_mono_float32(wav)


def _to_mono_float32(wav) -> np.ndarray:
    samples = np.asarray(wav)

    # Supertonic returns (1, n_samples) — a leading batch axis. Squeeze drops
    # any size-1 axis without touching a real channel axis. atleast_1d guards
    # the degenerate case where squeeze would produce a 0-d scalar.
    samples = np.atleast_1d(np.squeeze(samples))

    if samples.ndim > 1:
        # Still 2-D means genuine multichannel audio. Channels are the short
        # axis; average them rather than dropping one, so hard-panned content
        # does not lose half its volume.
        samples = samples.mean(axis=int(np.argmin(samples.shape)))

    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float32) / _INT16_SCALE
    else:
        samples = samples.astype(np.float32)

    return np.ascontiguousarray(samples)
