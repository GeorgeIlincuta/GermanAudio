import numpy as np
import pytest

from germanaudio.tts.supertonic import SupertonicEngine


class StubTTS:
    """Mimics supertonic.TTS closely enough to test the adapter's conversions."""

    def __init__(self, payload, **kwargs):
        self.payload = payload
        self.styles_requested: list[str] = []

    def get_voice_style(self, voice_name: str):
        self.styles_requested.append(voice_name)
        return f"style::{voice_name}"

    def synthesize(self, text, voice_style, lang):
        return self.payload, 1.0


def engine_with(payload) -> SupertonicEngine:
    return SupertonicEngine(tts=StubTTS(payload))


def test_returns_float32():
    engine = engine_with(np.array([0.1, -0.2], dtype=np.float64))

    result = engine.synthesize("das Fenster", "de", "M1")

    assert result.dtype == np.float32


def test_converts_int16_output_to_normalized_float():
    engine = engine_with(np.array([32767, -32768], dtype=np.int16))

    result = engine.synthesize("das Fenster", "de", "M1")

    assert result.dtype == np.float32
    assert result.max() <= 1.0
    assert result.min() >= -1.0
    assert result[0] == pytest.approx(1.0, abs=1e-3)


def test_flattens_stereo_to_mono():
    stereo = np.array([[0.4, 0.2], [0.6, 0.4], [0.8, 0.6]], dtype=np.float32)
    engine = engine_with(stereo)

    result = engine.synthesize("das Fenster", "de", "M1")

    assert result.ndim == 1
    assert len(result) == 3
    np.testing.assert_allclose(result, [0.3, 0.5, 0.7], atol=1e-6)


def test_squeezes_supertonic_batch_dimension():
    """Supertonic returns (1, n_samples); the batch axis must be dropped,
    not averaged over — averaging collapses the clip to one sample."""
    batched = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    engine = engine_with(batched)

    result = engine.synthesize("das Fenster", "de", "M1")

    assert result.ndim == 1
    assert len(result) == 4
    np.testing.assert_allclose(result, [0.1, 0.2, 0.3, 0.4], atol=1e-6)


def test_accepts_plain_list_output():
    engine = engine_with([0.1, 0.2, 0.3])

    result = engine.synthesize("das Fenster", "de", "M1")

    assert result.dtype == np.float32
    assert len(result) == 3


def test_caches_voice_styles_across_calls():
    """Building a style is not free; the same voice must not rebuild it."""
    stub = StubTTS(np.array([0.1], dtype=np.float32))
    engine = SupertonicEngine(tts=stub)

    engine.synthesize("eins", "de", "M1")
    engine.synthesize("zwei", "de", "M1")
    engine.synthesize("three", "en", "M1")

    assert stub.styles_requested == ["M1"]


def test_rejects_empty_text():
    engine = engine_with(np.array([0.1], dtype=np.float32))

    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", "de", "M1")


def test_engine_id_is_stable():
    assert SupertonicEngine(tts=StubTTS(np.array([0.1]))).engine_id == "supertonic-3"
