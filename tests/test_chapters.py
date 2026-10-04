import pytest
from mutagen.id3 import ID3

from germanaudio.assemble import build_timeline, encode_mp3, render, vocab_marks
from germanaudio.chapters import Chapter, mark_positions, write_tags
from germanaudio.config import PauseConfig, TextPauseConfig, VoiceConfig
from germanaudio.models import Entry
from germanaudio.reader import split_text, text_marks, text_timeline
from germanaudio.synth import ClipSpec, synthesize_clips

VOICES = VoiceConfig(de="M1", en="M1")
PAUSES = PauseConfig(after_word=1.5, after_translation=1.0, after_repeat_word=1.0,
                     after_repeat_translation=1.0, after_phrase=2.0)
RATE = 44100


def entries():
    return [
        Entry("das Fenster", "the window", "Mach das Fenster zu.", 2),
        Entry("der Schlüssel", "the key", "Wo ist der Schlüssel?", 3),
        Entry("die Tür", "the door", "Die Tür ist offen.", 4),
    ]


def synthesize(timeline, engine, cache_dir):
    clips = [item for item in timeline if isinstance(item, ClipSpec)]
    synthesize_clips(clips, engine, cache_dir)


def test_vocab_marks_start_every_entry_with_german_dash_english():
    marks = vocab_marks(entries())

    assert marks == {0: "das Fenster — the window", 10: "der Schlüssel — the key", 20: "die Tür — the door"}
    assert len(build_timeline(entries(), VOICES, PAUSES)) == 30


def test_mark_positions_are_exact_sample_offsets_in_the_rendered_track(tmp_path, fake_engine):
    timeline = build_timeline(entries(), VOICES, PAUSES)
    synthesize(timeline, fake_engine, tmp_path)
    samples = render(timeline, tmp_path, fake_engine.engine_id, RATE, target_peak=1.0)

    chapters, total = mark_positions(timeline, vocab_marks(entries()), tmp_path, fake_engine.engine_id, RATE)

    assert total == len(samples)
    assert [c.title for c in chapters] == ["das Fenster — the window", "der Schlüssel — the key", "die Tür — the door"]
    assert chapters[0].start_sample == 0
    # each chapter starts right after the previous entry's 2.0 s closing pause, i.e. where the next word's audio begins
    for chapter in chapters[1:]:
        before = samples[chapter.start_sample - int(2.0 * RATE): chapter.start_sample]
        assert not before.any()
        assert samples[chapter.start_sample: chapter.start_sample + 200].any()


def test_text_marks_point_at_spoken_headings(tmp_path, fake_engine):
    paragraphs = split_text("Der Titel.\n\n# Der Anfang — The beginning\nEs war einmal.\n\n# Das Ende\nAus.")
    pauses = TextPauseConfig()
    timeline = text_timeline(paragraphs, "M1", pauses)

    marks = text_marks(paragraphs)

    assert marks == {2: "Der Anfang — The beginning", 6: "Das Ende"}
    assert timeline[2] == ClipSpec("Der Anfang.", "de", "M1")
    assert timeline[6] == ClipSpec("Das Ende.", "de", "M1")


def tag_file(tmp_path, chapters, total=10 * RATE):
    path = tmp_path / "t.mp3"
    encode_mp3(__import__("numpy").zeros(total, dtype="float32"), RATE, path, 64)
    write_tags(path, "B1 Vokabeln", chapters, total, RATE)
    return path


def test_write_tags_adds_title_and_chapters_ending_at_the_next_start(tmp_path):
    path = tag_file(tmp_path, [Chapter("eins — one", 0), Chapter("zwei — two", 3 * RATE)])

    tags = ID3(path)
    assert tags["TIT2"].text == ["B1 Vokabeln"]
    chaps = sorted(tags.getall("CHAP"), key=lambda c: c.start_time)
    assert [(c.start_time, c.end_time, c.sub_frames["TIT2"].text[0]) for c in chaps] == [
        (0, 3000, "eins — one"),
        (3000, 10000, "zwei — two"),
    ]


def test_write_tags_nests_the_table_of_contents_beyond_255_chapters(tmp_path):
    chapters = [Chapter(f"Wort {i}", i * RATE // 100) for i in range(600)]

    tags = ID3(tag_file(tmp_path, chapters))

    assert len(tags.getall("CHAP")) == 600
    tocs = {toc.element_id: toc for toc in tags.getall("CTOC")}
    assert tocs["toc"].child_element_ids == ["toc0", "toc1", "toc2"]
    assert sum(len(tocs[f"toc{i}"].child_element_ids) for i in range(3)) == 600


def test_write_tags_without_chapters_writes_only_the_title(tmp_path):
    tags = ID3(tag_file(tmp_path, []))

    assert tags["TIT2"].text == ["B1 Vokabeln"]
    assert tags.getall("CHAP") == []


def test_write_tags_keeps_the_audio_playable_and_untouched(tmp_path):
    path = tmp_path / "t.mp3"
    encode_mp3(__import__("numpy").zeros(RATE, dtype="float32"), RATE, path, 64)
    audio = path.read_bytes()

    write_tags(path, "T", [Chapter("a", 0)], RATE, RATE)

    assert path.read_bytes().endswith(audio)
