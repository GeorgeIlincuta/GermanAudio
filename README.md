# germanaudio

Turns German text into MP3 listening tracks, using the Supertonic text-to-speech model locally. It has two modes:

- **Vocabulary track** (default): from a list of words it builds a drill track. For each entry you hear the German
  word, its English translation, both again, then an example phrase. Each part is followed by a pause, and the pause
  after the first German word is your time to recall the meaning.
- **Read aloud** (`--text`): reads any German text file (a story, an article, a dialogue) from start to finish with
  the German voice.

## Setup

Requires Python 3 (developed on 3.14).

```
pip install -r requirements.txt
```

The Supertonic model downloads automatically on the first run.

## Vocabulary track

### 1. Make your vocabulary file

Open a Claude chat, paste the prompt from `prompts/generate-vocab.md`, and work in batches of 50–100 rows. Paste each
code block into `vocab.tsv` at the project root, keeping only the header from the first batch.

```
german	english	phrase
das Fenster	the window	Bitte mach das Fenster zu, es ist kalt draußen.
```

The file needs three tab-separated columns and the header row. Pipes (`|`) also work as separators, because tabs
sometimes turn into spaces when you copy text out of a chat window. `sample.tsv` is a working eight-row example.

### 2. Check it before spending any time

```
python -m germanaudio vocab.tsv --check
```

This validates the file and exits without making any audio. It lists every bad line with its line number in one go,
so a batch that pasted wrong takes one round of fixes.

### 3. Build the track

```
python -m germanaudio vocab.tsv -o out/german.mp3
```

For 500 words expect roughly 1,500 clips, a few minutes of synthesis, and a 2-hour, 60 MB MP3.

### Useful in practice

Preview your settings before committing to the full run:

```
python -m germanaudio vocab.tsv --limit 10 -o out/preview.mp3
```

Listen, adjust the pauses and run it again. The ten clips it already made are cached, so the next run only makes
what's new.

The pause lengths are in `PauseConfig` in `germanaudio/config.py`:

```python
after_word = 1.5              # the recall gap, the one you actually use
after_translation = 1.0
after_repeat_word = 1.0
after_repeat_translation = 1.0
after_phrase = 2.0            # separates entries
```

Changing a pause is free. Pauses are generated rather than synthesized, so a re-run reuses every cached clip and
only re-stitches the track.

Changing a word or a voice isn't free. The cache is keyed on the text, so editing one phrase re-synthesizes that one
clip, and changing a voice in `config.py` re-synthesizes everything.

Interrupted runs resume: if you press Ctrl-C at entry 400, the next run picks up at 400.

## Read a German text aloud (`--text`)

This mode gives you one MP3 of the text read out loud, with no translations and no drill.

### 1. Make your text file

Paste the text into a new Notepad file and save it as a `.txt`, for example `story.txt` at the project root. The file
must be UTF-8 for umlauts and ß to work. Notepad on Windows 11 saves as UTF-8 by default; if you use Save As, keep
Encoding set to UTF-8.

```
Es war einmal ein kleiner Fuchs, der im Wald lebte. Er war sehr neugierig!
Am 3. Mai ging er zum ersten Mal in die Stadt.

„Wo bin ich hier?“, fragte er. Niemand antwortete.
```

How the text is read:

- Each sentence (ending in `.` `!` `?` or `…`) is spoken separately, with a short pause after it.
- An empty line starts a new paragraph, which gets a longer pause.
- Line breaks without an empty line are ignored, so text pasted from a PDF or web page with hard line wraps still
  reads as normal sentences.
- There's no header and no columns.

### 2. Check it

```
python -m germanaudio story.txt --text --check
```

This prints something like `loaded 7 sentences in 2 paragraphs` and exits without making any audio. If the count
looks wrong (say, one paragraph when you expected three), check the empty lines in the file.

### 3. Preview the first few sentences

```
python -m germanaudio story.txt --text --limit 5 -o out/preview.mp3
```

Listen to check the voice and pauses before you do a long text. The five clips are cached, so the full run reuses
them.

### 4. Build the full audio

```
python -m germanaudio story.txt --text -o out/story.mp3
```

Always give `-o` a name of its own. Without it the output goes to `out/german.mp3` and overwrites your vocabulary
track. Interrupted runs resume, and editing one sentence re-synthesizes only that sentence.

The pauses are in `TextPauseConfig` in `germanaudio/config.py`:

```python
after_sentence = 0.6          # between sentences
after_paragraph = 1.5         # after each paragraph
```

As with the vocabulary pauses, changing these is free.

Don't forget `--text`. Without it the file is treated as a vocabulary list, and `--check` reports that the header is
wrong.

## Known problems

**Some abbreviations add a pause.** Text is split into sentences at `.` `!` `?` `…` followed by a space. Two common
false ends are handled: a period before a lowercase word (`ca. zehn`, `bzw. auch`) and a period after a number
(`am 3. Mai`) don't split. But `z. B. Die…`, `Dr. Müller` or `Nr. 5` do, so you hear a 0.6 s pause mid-sentence.
It's harmless; if it bothers you, write `zum Beispiel` or `Doktor` in the text.

**Supertonic rejects four German quote marks: `„ ‚ › ‹`.** They are swapped for `“ ‘ ’ ‘` automatically before
synthesis, so you don't need to edit your text, and since quote marks aren't spoken it sounds the same. These
characters were tested and are accepted: `“ ” " ' ‘ ’ » « … – — € % &`, umlauts and `ß`. Anything more unusual may
still fail. A failure is reported per sentence and nothing else is lost: fix the character, run again, and only that
clip is synthesized.

**Error messages may show German characters as `�`.** That comes from the Windows console's codepage, not your
file, which is fine.

**Memory.** The whole track is held in memory before encoding. At 500 entries (about two hours) the peak is roughly
2 GB. That's fine on a normal machine, but if it struggles, split your list in two and build two files.

**Loudness.** The sample track peaked at 0.52, so about 6 dB of volume is unused. There's no normalization step yet.

## Tests

```
python -m pytest
```
