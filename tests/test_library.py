from pathlib import Path
from types import SimpleNamespace

from mutagen.id3 import ID3, TIT2

from germanaudio.library import library_path, numbered_title, publish, strip_number


def _track(path: Path, title: str, size: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    tag = ID3()
    tag.add(TIT2(encoding=3, text=[title]))
    tag.save(path)
    return path


def test_strip_number():
    assert strip_number("07 · Lager Vokabeln") == "Lager Vokabeln"
    assert strip_number("Lager Vokabeln") == "Lager Vokabeln"
    assert strip_number("B1 Vokabeln") == "B1 Vokabeln"


def test_a_new_track_gets_the_next_number_in_its_category(tmp_path):
    _track(tmp_path / "Wortschatz/B1/b1.mp3", "01 · B1 Vokabeln")
    _track(tmp_path / "Wortschatz/Lager/lager-1.mp3", "07 · Lager Vokabeln")
    _track(tmp_path / "Geschichten/x.mp3", "12 · Andere Kategorie")
    assert numbered_title("Zucker Vokabeln", "Wortschatz", "zucker.mp3", tmp_path) == "08 · Zucker Vokabeln"


def test_a_published_track_keeps_its_number(tmp_path):
    _track(tmp_path / "Wortschatz/B1/b1.mp3", "01 · B1 Vokabeln")
    _track(tmp_path / "Wortschatz/Lager/lager-1.mp3", "07 · Lager Vokabeln")
    assert numbered_title("B1 Vokabeln", "Wortschatz", "b1.mp3", tmp_path) == "01 · B1 Vokabeln"


def test_a_numbered_title_is_kept_as_given(tmp_path):
    assert numbered_title("03 · Eigene Nummer", "Wortschatz", "x.mp3", tmp_path) == "03 · Eigene Nummer"


def test_the_first_track_of_a_category_is_01(tmp_path):
    assert numbered_title("Der erste Tag", "Geschichte", "neu.mp3", tmp_path) == "01 · Der erste Tag"


def test_no_library_folder_means_no_number(tmp_path):
    assert numbered_title("Der erste Tag", "Geschichte", "neu.mp3", tmp_path / "missing") == "Der erste Tag"


def test_library_path_per_category(tmp_path):
    assert library_path("Wortschatz", "09 · Zucker Vokabeln", "zucker.mp3", tmp_path) == Path("Wortschatz/Zucker/zucker.mp3")
    assert library_path("Geschichte", "06 · X", "x_geschichte.mp3", tmp_path) == Path("Geschichten/x_geschichte.mp3")
    assert library_path("Artikel", "02 · Y", "y.mp3", tmp_path) == Path("Artikel/y.mp3")
    assert library_path("Enzyklopädie", "17 · Z", "z.mp3", tmp_path) == Path("Enzyklopädie/z.mp3")
    assert library_path("Leitfaden", "07 · G", "g.mp3", tmp_path) == Path("Leitfaden/g.mp3")


def test_library_path_reuses_an_existing_folder(tmp_path):
    _track(tmp_path / "Wortschatz/Arbeit/arbeit-saetze.mp3", "06 · Arbeit Sätze")
    assert library_path("Wortschatz", "06 · Neuer Name", "arbeit-saetze.mp3", tmp_path) == Path("Wortschatz/Arbeit/arbeit-saetze.mp3")


class FakeAdb:
    def __init__(self, connected=True, sizes=None):
        self.connected, self.sizes, self.calls = connected, sizes or {}, []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[-1] == "devices":
            out = "List of devices attached\nserial1\tdevice\n" if self.connected else "List of devices attached\n"
        elif "push" in cmd:
            self.sizes[cmd[-1]] = Path(cmd[-2]).stat().st_size
            out = "1 file pushed"
        else:  # shell stat -c %s '<path>'
            path = cmd[-1].split("'")[1]
            out = f"{self.sizes.get(path, -1)}\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")


def test_publish_copies_to_the_library_and_the_phone(tmp_path):
    track = _track(tmp_path / "out/zucker.mp3", "09 · Zucker Vokabeln", size=100)
    root = tmp_path / "MTL"
    adb = FakeAdb()
    ok, report = publish(track, "Wortschatz", "09 · Zucker Vokabeln", root, adb_path=tmp_path / "adb.exe",
                         serial="serial1", phone_root="/sd/MTL", run=adb)
    assert ok
    assert (root / "Wortschatz/Zucker/zucker.mp3").read_bytes() == track.read_bytes()
    assert ["push", str(track), "/sd/MTL/Wortschatz/Zucker/zucker.mp3"] == adb.calls[1][-3:]
    assert any("phone" in line for line in report)


def test_publish_skips_a_disconnected_phone(tmp_path):
    track = _track(tmp_path / "out/a.mp3", "01 · A")
    ok, report = publish(track, "Artikel", "01 · A", tmp_path / "MTL", adb_path=tmp_path / "adb.exe",
                         serial="serial1", phone_root="/sd/MTL", run=FakeAdb(connected=False))
    assert ok
    assert (tmp_path / "MTL/Artikel/a.mp3").exists()
    assert any("not connected" in line for line in report)


def test_publish_reports_a_size_mismatch_on_the_phone(tmp_path):
    track = _track(tmp_path / "out/a.mp3", "01 · A")
    adb = FakeAdb()
    adb_push = adb.__call__

    def broken(cmd, **kwargs):
        result = adb_push(cmd, **kwargs)
        if "push" in cmd:
            adb.sizes[cmd[-1]] = 1
        return result

    ok, report = publish(track, "Artikel", "01 · A", tmp_path / "MTL", adb_path=tmp_path / "adb.exe",
                         serial="serial1", phone_root="/sd/MTL", run=broken)
    assert not ok
    assert any("size" in line for line in report)
