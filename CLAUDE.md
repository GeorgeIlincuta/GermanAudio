# GermanAudio — notes for Claude

Turns German vocabulary lists and stories into MP3 listening tracks (Supertonic TTS, local). The MP3s are played in
the user's own Android app **Hörplayer** (repo: `C:\LOCAL FILES\Claude Code\Kotlin\AudioPlayer`, see its
`docs/PROJECT-STATUS.md`). The user learns German (goal B2) and Biology/Chemistry for the MTL Ausbildung.
Talk to the user in English; all learning content is German.

## The Hörplayer metadata contract (every track must follow it)

The generator writes this automatically (`germanaudio/chapters.py`, called by the CLI after encoding). Don't
hand-edit tags; fix the input and rebuild.

- **ID3v2.3 tag** in front of the audio: `TIT2` = file title shown on the library card.
- **One `CHAP` frame per section**, start/end in ms, with an embedded `TIT2` = section title. Ends = next start.
  More than 255 chapters → a top-level `CTOC` pointing to sub-`CTOC`s (one CTOC holds at most 255 children).
- **Section titles use `German — English`**: space, em dash U+2014, space. The app shows the German part big and
  the English part underneath. A hyphen or en dash is NOT a separator.
- **Vocabulary track:** one chapter per entry, title `{german} — {english}` (e.g. `das Fenster — the window`).
  The app labels these "Abschnitte".
- **Story (`--text`):** one chapter per heading line `# Deutsch — English` (e.g. `# Der erste Tag — The first day`).
  Only the German part is spoken (`Der erste Tag.` — a period is added if missing). No chapter numbers in headings
  (the app shows "Kapitel 3 von 14" itself). A new chapter every 3–6 minutes (~350–650 words). The app labels these
  "Kapitel".
- Chapter times are computed from cached clip lengths + configured pauses, identical to `render`; they're exact.
- **Read-along text (stories only):** one `SYLT` frame (UTF-16, `deu`, ms timestamps, type lyrics) with one entry per
  spoken sentence: the sentence as written, a leading line break on the first sentence of each paragraph, and
  `# Deutsch — English` for headings (same time as their `CHAP`). The Hörplayer highlights the sentence being spoken.
  Vocabulary tracks have no `SYLT`.
- **Artikel (`--article`):** one file = the article's vocabulary drill, 3 s, then the article read aloud (same rules as
  a story). Extra frames: `TXXX` `Artikel-Start` (ms where the article part begins = its first `CHAP`), `TPE1` author,
  `TDRC` published, `WOAS` source URL, `TXXX` `Rubrik` (category), `TXXX` `Studie` (study citation); a missing
  front-matter key writes no frame. `SYLT` covers the article part only and holds the **shown** text (see overrides).
  The Hörplayer (1.4+) shows "Wort 12 von 60" / "Kapitel 2 von 4" and the metadata at the top of the Text tab.
- **Spoken overrides `{shown|spoken}`** (stories and articles): `{37|siebenunddreißig}` — the voice reads the right
  side, the read-along shows the left. Use for digits, abbreviations, symbols. `--check` warns about digits outside one.
- **Genre = file type:** one `TCON` frame (UTF-16): `Wortschatz` for vocabulary, `Geschichte` for `--text` stories,
  `Artikel` for `--article`.
  The Hörplayer shows its type icon from it. Build and `--retag` both write it.
- **Retag** (adds/refreshes title, chapters and text without re-synthesizing; the audio is verified unchanged):
  `python -m germanaudio <story>.txt --text --retag -o out/<story>.mp3`. It refuses (and changes nothing) if a
  clip is missing from the cache or the text no longer matches the audio.
  The audio bytes are never changed by tagging.

## Titles (TIT2) — conventions in use

| Input | Output | Title |
|---|---|---|
| `b1.tsv` | `out/b1.mp3` | `B1 Vokabeln` |
| `chemie.tsv` | `out/chemie.mp3` | `Chemie Vokabeln` |
| `mikrobiom.tsv` | `out/mikrobiom.mp3` | `Mikrobiom Vokabeln` |
| `b1_geschichte.txt` | `out/b1_geschichte.mp3` | first sentence: `Ioanas neues Leben` |
| `chemie_geschichte.txt` | `out/chemie_geschichte.mp3` | `Lenas Jahr im Labor` |
| `mikrobiom_geschichte.txt` | `out/mikrobiom_geschichte.mp3` | `Jonas und die unsichtbaren Helfer` |
| `schmerz.tsv` | `out/schmerz.mp3` | `Schmerz Vokabeln` |
| `schmerz_geschichte.txt` | `out/schmerz_geschichte.mp3` | `Bewegung statt Tabletten` |

Vocab tracks: always pass `--title "<Thema> Vokabeln"` (the default would be the bare file name). Stories: the
default (first sentence of the text, without its period) is right; the first line of a story is its title.

## Workflow for a new topic

1. **Vocabulary list:** prompt `prompts/prompt-vocab.txt` (or `generate-vocab.md`) in a Claude chat → `<topic>.tsv`
   (tab-separated `german  english  phrase`, header row). Check: `python -m germanaudio <topic>.tsv --check`.
2. **Build vocab first** (its sentences are then reused by the story):
   `python -m germanaudio <topic>.tsv -o out/<topic>.mp3 --title "<Topic> Vokabeln"`
3. **Story:** prompt `prompts/prompt-story.txt` (already asks for `#` headings) → `<topic>_geschichte.txt`, UTF-8.
   Check: `python -m germanaudio <topic>_geschichte.txt --text --check` → must report "… and N headings".
4. **Build story:** `python -m germanaudio <topic>_geschichte.txt --text -o out/<topic>_geschichte.mp3`
   Always give `-o` its own name (default would overwrite `out/german.mp3`).
5. **Verify** (prints title + chapter count + first chapter):
   ```
   python -c "from mutagen.id3 import ID3; t=ID3('out/<file>.mp3'); c=sorted(t.getall('CHAP'),key=lambda x:x.start_time); print(t['TIT2'].text, len(c), c[0].sub_frames['TIT2'].text if c else '-')"
   ```
   The CLI also prints `title '…', N chapters` at the end of a build.
6. Commit the new `.tsv` / `.txt` inputs (they're the only way to rebuild a track; `out/` and `cache/` are ignored).

Set `PYTHONIOENCODING=utf-8` in Git Bash so umlauts print correctly. Before replacing an existing MP3 that matters,
copy it to `out/<name>.original.mp3`.

## Workflow for an article (wissenschaft.de)

1. The user saves the article with the WissenschaftExtractor Chrome extension (1.1+,
   `C:\LOCAL FILES\Claude Code\Chrome Extensions\WissenschaftExtractor`) → `Downloads\Wissenschaft\<slug>-<date>.md`
   in the Artikel source format (front matter, empty `## Wortschatz`, `## Artikel` with German-only `#` headings).
2. Copy it to `artikel/<topic>.md` (leave the download untouched). Then Claude: removes anything that shouldn't be
   heard; fills `## Wortschatz` with ~60 B2+ words (`german | english | phrase`, new B1 example sentences, same rules
   as `prompts/prompt-vocab.txt`); adds the English half of every heading; wraps digits/abbreviations in
   `{shown|spoken}`; replaces „“ with »«.
3. The user reviews the word list.
4. `python -m germanaudio artikel/<topic>.md --article --check` → `loaded N entries and … headings`.
5. `python -m germanaudio artikel/<topic>.md --article -o out/<topic>.mp3` (title = front-matter `title`).
6. Push to `MTL/Artikel/<topic>.mp3`, copy to `C:\LOCAL FILES\Study\German`, commit `artikel/<topic>.md`.

## Getting tracks onto the phone

Phone: Samsung Galaxy S9+ (Android 10), adb serial `21fcf6986b0c7ece`,
adb = `C:\Users\ilinc\AppData\Local\Android\Sdk\platform-tools\adb.exe`. In Git Bash set `MSYS_NO_PATHCONV=1`.
All tracks live under the **SD-card folder `MTL`**: `/storage/3838-6430/MTL/` (the Hörplayer watches this folder and
rescans it every time its library opens — new files appear, changed files are re-read automatically). Layout:
vocabulary in `MTL/Wortschatz/<Thema>/<topic>.mp3` (e.g. `Wortschatz/Schmerz/schmerz.mp3`), stories in
`MTL/Geschichten/<topic>_geschichte.mp3`, articles in `MTL/Artikel/<topic>.mp3`. (StudyAudio tracks: `MTL/Leitfaden/` guides, `MTL/Enzyklopädie/` encyclopedias.)

```
adb push out/<topic>.mp3 /storage/3838-6430/MTL/Wortschatz/<Thema>/<topic>.mp3
adb push out/<topic>_geschichte.mp3 /storage/3838-6430/MTL/Geschichten/<topic>_geschichte.mp3
```

If `adb devices` lists nothing (USB debugging off), copy over MTP instead: the phone shows up in Windows as the
portable device `Senex`, SD card = `Card`; PowerShell `Shell.Application` → `NameSpace(17)` → `Senex` → `Card` → `MTL`,
then `CopyHere`. Compare `System.Size` with the local file afterwards. Also copy finished tracks to
`C:\LOCAL FILES\Study\German`.

Overwrite in place rather than adding a second copy (keeps the user's place in the app). Only finished, tagged
files go there — no `.original` backups.

## Code map and conventions

- `loader.py` (TSV, `parse_vocab_rows`) · `reader.py` (text, `Heading`, `text_marks`, `story_title`, overrides) ·
  `article.py` (Artikel source file) · `synth.py` (content-hash clip
  cache in `cache/audio`) · `assemble.py` (timeline, `vocab_marks`, render, MP3 encode) · `chapters.py`
  (`mark_positions`, `write_tags`) · `cli.py` · tests in `tests/` (`python -m pytest -q`, no real model needed).
- Changing pauses or titles is free (re-stitch from cache); changing text or voice re-synthesizes those clips.
- Work test-first; keep `build_timeline` / `text_timeline` and their marks functions in step (marks are timeline
  indices).
