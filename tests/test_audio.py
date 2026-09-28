import numpy as np
import pytest

from germanaudio.audio import read_wav, silence, write_wav


def test_round_trip_preserves_samples(tmp_path):
    original = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    path = tmp_path / "clip.wav"

    write_wav(path, original, 44100)
    restored, rate = read_wav(path)

    assert rate == 44100
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, original, atol=1e-4)


def test_round_trip_preserves_full_scale_extremes(tmp_path):
    original = np.array([1.0, -1.0], dtype=np.float32)
    path = tmp_path / "clip.wav"

    write_wav(path, original, 44100)
    restored, _ = read_wav(path)

    np.testing.assert_allclose(restored, original, atol=1e-4)


def test_clips_out_of_range_samples(tmp_path):
    """Values beyond +/-1 must clamp, not wrap around into loud noise."""
    original = np.array([2.0, -2.0], dtype=np.float32)
    path = tmp_path / "clip.wav"

    write_wav(path, original, 44100)
    restored, _ = read_wav(path)

    assert restored.max() <= 1.0
    assert restored.min() >= -1.0


def test_silence_has_correct_length_and_is_quiet():
    samples = silence(1.5, 44100)

    assert len(samples) == 66150
    assert samples.dtype == np.float32
    assert np.all(samples == 0.0)


def test_silence_of_zero_seconds_is_empty():
    assert len(silence(0.0, 44100)) == 0


def test_read_wav_rejects_stereo(tmp_path):
    import wave

    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00\x00\x00")

    with pytest.raises(ValueError, match="mono"):
        read_wav(path)
