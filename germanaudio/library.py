"""The Hörplayer library: numbered titles and publishing finished tracks.

The published library folder (`C:\\LOCAL FILES\\Study\\MTL`, the folder the desktop Hörplayer watches, mirrored on the
phone's SD card under `MTL`) is the record of every track's number. A title is "NN · Short title", numbered in build
order within its category folder: a track already in the library keeps its number, a new one gets the next.
GermanAudio and StudyAudio both read that folder, so tracks of either repo in one category (Geschichten) share one
numbering. This module is the same in both repos; keep them in step.
"""

import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

from mutagen.id3 import ID3, ID3NoHeaderError

LIBRARY_ROOT = Path(r"C:\LOCAL FILES\Study\MTL")
PHONE_ROOT = "/storage/3838-6430/MTL"
PHONE_SERIAL = "21fcf6986b0c7ece"
ADB = Path(r"C:\Users\ilinc\AppData\Local\Android\Sdk\platform-tools\adb.exe")

# ID3 genre (the file type) -> its folder in the library.
CATEGORY_FOLDERS = {
    "Wortschatz": "Wortschatz",
    "Geschichte": "Geschichten",
    "Artikel": "Artikel",
    "Leitfaden": "Leitfaden",
    "Enzyklopädie": "Enzyklopädie",
}

_NUMBERED = re.compile(r"^(\d{2,}) · (.+)$")


def strip_number(title: str) -> str:
    """"07 · Lager Vokabeln" -> "Lager Vokabeln"; an unnumbered title is returned unchanged."""
    match = _NUMBERED.match(title)
    return match.group(2) if match else title


def _title_of(path: Path) -> str | None:
    try:
        frame = ID3(path).get("TIT2")
    except (ID3NoHeaderError, OSError):
        return None
    return str(frame.text[0]) if frame and frame.text else None


def _category_tracks(genre: str, root: Path) -> list[Path]:
    folder = Path(root) / CATEGORY_FOLDERS.get(genre, genre)
    return sorted(folder.rglob("*.mp3")) if folder.is_dir() else []


def numbered_title(title: str, genre: str, file_name: str, root: Path = LIBRARY_ROOT) -> str:
    """The title with its category number: kept if `title` is already numbered, reused if `file_name` is already in
    the library, else one more than the highest number in the category (01 for the first). Without a library
    folder the title is returned unchanged."""
    if _NUMBERED.match(title) or not Path(root).is_dir():
        return title  # numbered by hand, or no library on this machine to number from
    numbers = {}
    for path in _category_tracks(genre, root):
        match = _NUMBERED.match(_title_of(path) or "")
        if match:
            numbers[path.name] = int(match.group(1))
    number = numbers.get(file_name, max(numbers.values(), default=0) + 1)
    return f"{number:02d} · {title}"


def library_path(genre: str, title: str, file_name: str, root: Path = LIBRARY_ROOT) -> Path:
    """Where a track goes, relative to the library root: `Wortschatz/<Thema>/<file>` (Thema = first word of the
    title, or the folder the file is already in) or `<category folder>/<file>`."""
    folder = CATEGORY_FOLDERS.get(genre, genre)
    for path in _category_tracks(genre, root):
        if path.name == file_name:
            return path.relative_to(root)
    if genre == "Wortschatz":
        return Path(folder, strip_number(title).split()[0], file_name)
    return Path(folder, file_name)


def publish(track: Path, genre: str, title: str, root: Path = LIBRARY_ROOT, *, adb_path: Path = ADB,
            serial: str = PHONE_SERIAL, phone_root: str = PHONE_ROOT, run=subprocess.run) -> tuple[bool, list[str]]:
    """Copy a finished track into the library folder and onto the phone (when it's connected over adb), overwriting
    in place so the Hörplayer keeps the listening position. Returns (all copies verified, report lines)."""
    track = Path(track)
    relative = library_path(genre, title, track.name, root)
    size = track.stat().st_size
    report, ok = [], True

    target = Path(root) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(track, target)
    if target.stat().st_size == size:
        report.append(f"library: {target}")
    else:
        ok = False
        report.append(f"library: {target} — size differs after copying")

    def adb(*args: str):
        return run([str(adb_path), "-s", serial, *args], capture_output=True, text=True, encoding="utf-8")

    try:
        devices = run([str(adb_path), "devices"], capture_output=True, text=True, encoding="utf-8").stdout
    except OSError:
        devices = ""
    if f"{serial}\tdevice" not in devices:
        report.append("phone: not connected (USB debugging off?) — skipped; copy it later")
        return ok, report

    remote = str(PurePosixPath(phone_root, *relative.parts))
    pushed = adb("push", str(track), remote)
    remote_size = adb("shell", f"stat -c %s '{remote}'").stdout.strip()
    if pushed.returncode == 0 and remote_size == str(size):
        report.append(f"phone: {remote}")
    else:
        ok = False
        report.append(f"phone: {remote} — push failed or size differs ({remote_size or pushed.stderr.strip()})")
    return ok, report
