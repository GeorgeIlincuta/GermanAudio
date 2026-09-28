import numpy as np
import pytest

from germanaudio.assemble import Silence, build_timeline, encode_mp3, render
from germanaudio.config import PauseConfig, VoiceConfig
from germanaudio.models import Entry
from germanaudio.synth import ClipSpec, synthesize_all

VOICES = VoiceConfig(de="M1", en="F1")
PAUSES = PauseConfig()


def entry(german="das Fenster", english="the window", phrase="Mach das Fenster zu.", line=2):
    return Entry(german, english, phrase, line)


def test_timeline_has_five_clips_and_five_pauses_per_entry():
    timeline = build_timeline([entry()], VOICES, PAUSES)

    assert len(timeline) == 10
    assert sum(1 for item in timeline if isinstance(item, ClipSpec)) == 5
    assert sum(1 for item in timeline if isinstance(item, Silence)) == 5


def test_timeline_follows_the_de_en_de_en_phrase_pattern():
    timeline = build_timeline([entry()], VOICES, PAUSES)
    clips = [item for item in timeline if isinstance(item, ClipSpec)]

    assert [clip.text for clip in clips] == [
        "das Fenster",
        "the window",
        "das Fenster",
        "the window",
        "Mach das Fenster zu.",
    ]
    assert [clip.language for clip in clips] == ["de", "en", "de", "en", "de"]


def test_repeated_slots_reference_the_identical_clip_spec():
    """Slots 3 and 4 must be the same specs as 1 and 2, or the cache is missed."""
    timeline = build_timeline([entry()], VOICES, PAUSES)
    clips = [item for item in timeline if isinstance(item, ClipSpec)]

    assert clips[0] == clips[2]
    assert clips[1] == clips[3]


def test_timeline_alternates_clip_then_silence():
    timeline = build_timeline([entry()], VOICES, PAUSES)

    for index, item in enumerate(timeline):
        expected = ClipSpec if index % 2 == 0 else Silence
        assert isinstance(item, expected)


def test_pause_durations_come_from_config():
    pauses = PauseConfig(
        after_word=3.0,
        after_translation=0.5,
        after_repeat_word=0.6,
        after_repeat_translation=0.7,
        after_phrase=4.0,
    )

    timeline = build_timeline([entry()], VOICES, pauses)
    gaps = [item.seconds for item in timeline if isinstance(item, Silence)]

    assert gaps == [3.0, 0.5, 0.6, 0.7, 4.0]


def test_entries_appear_in_file_order():
    entries = [
        entry(german="eins", english="one", phrase="Eins ist die erste Zahl."),
        entry(german="zwei", english="two", phrase="Zwei kommt nach eins."),
    ]

    timeline = build_timeline(entries, VOICES, PAUSES)
    clips = [item for item in timeline if isinstance(item, ClipSpec)]

    assert clips[0].text == "eins"
    assert clips[5].text == "zwei"


def test_empty_entry_list_produces_empty_timeline():
    assert build_timeline([], VOICES, PAUSES) == []


def test_render_concatenates_clips_and_silence(tmp_path, fake_engine):
    entries = [entry()]
    synthesize_all(entries, fake_engine, tmp_path, VOICES)
    timeline = build_timeline(entries, VOICES, PAUSES)

    samples = render(timeline, tmp_path, fake_engine.engine_id, fake_engine.sample_rate)

    assert samples.dtype == np.float32
    expected_silence = sum(item.seconds for item in timeline if isinstance(item, Silence))
    assert len(samples) > int(expected_silence * fake_engine.sample_rate)


def test_render_total_duration_equals_sum_of_parts(tmp_path, fake_engine):
    from germanaudio.audio import read_wav
    from germanaudio.synth import clip_path

    entries = [entry()]
    synthesize_all(entries, fake_engine, tmp_path, VOICES)
    timeline = build_timeline(entries, VOICES, PAUSES)

    expected = 0
    for item in timeline:
        if isinstance(item, Silence):
            expected += int(round(item.seconds * fake_engine.sample_rate))
        else:
            clip, _ = read_wav(clip_path(tmp_path, item, fake_engine.engine_id))
            expected += len(clip)

    samples = render(timeline, tmp_path, fake_engine.engine_id, fake_engine.sample_rate)

    assert len(samples) == expected


def test_render_reports_a_missing_clip_by_text(tmp_path, fake_engine):
    timeline = build_timeline([entry()], VOICES, PAUSES)

    with pytest.raises(FileNotFoundError, match="das Fenster"):
        render(timeline, tmp_path, fake_engine.engine_id, fake_engine.sample_rate)


def test_encode_mp3_writes_a_real_mp3(tmp_path):
    samples = (0.3 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)).astype(np.float32)
    path = tmp_path / "track.mp3"

    encode_mp3(samples, 44100, path, bitrate=64)

    assert path.exists()
    assert path.stat().st_size > 1000
    # An MP3 frame sync is eleven set bits: 0xFF then the top three bits of the
    # next byte. Checking only the first byte would also accept a JPEG (0xFFD8).
    head = path.read_bytes()[:3]
    assert head == b"ID3" or (head[0] == 0xFF and head[1] & 0xE0 == 0xE0)


def test_encode_mp3_creates_the_output_directory(tmp_path):
    samples = np.zeros(44100, dtype=np.float32)
    path = tmp_path / "nested" / "dir" / "track.mp3"

    encode_mp3(samples, 44100, path, bitrate=64)

    assert path.exists()
