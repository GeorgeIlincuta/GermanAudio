# Prompts for generating `vocab.tsv`

Paste one of these into a Claude chat. Copy the code block it returns into
`vocab.tsv`. Work in batches of 50–100 rows — asking for all 500 at once
risks a truncated response, and a truncated row is easy to miss.

For the second and later batches, add: *"Continue with the next 50. Do not
repeat any word from the previous batches."* Append each code block to the
file and delete the duplicated header rows, keeping only the one at the top.

---

## Prompt A — you have the words, you need translations and phrases

> I'm building a German listening track and need a data file in an exact
> format. Output **one code block and nothing else** — no commentary before
> or after it, no numbering, no bullet points, no blank lines inside.
>
> The format is three tab-separated columns with this exact header row:
>
> ```
> german	english	phrase
> ```
>
> Then one line per word, tab-separated, in this order:
>
> 1. **german** — the German word. Give nouns with their definite article
>    (`das Fenster`, not `Fenster`). Give verbs in the infinitive.
> 2. **english** — a short English translation, two or three words at most.
>    No parenthetical notes, no alternatives separated by slashes.
> 3. **phrase** — one natural German sentence that uses the word, 8 to 15
>    words, at CEFR level A2–B1. Everyday, concrete situations. The word may
>    appear inflected or declined — that is expected and correct. Do not
>    include an English translation of the sentence.
>
> Rules:
>
> - Tab characters between columns. Never commas — the sentences will
>   contain commas and they must not be treated as separators.
> - Exactly three columns per line. No trailing tab.
> - Every field filled. No empty cells, no placeholders, no `—`.
> - No duplicate entries in the `german` column.
> - Plain text only inside the block: no quotation marks wrapping fields, no
>   markdown, no escaping.
>
> Example of correct output:
>
> ```
> german	english	phrase
> das Fenster	the window	Bitte mach das Fenster zu, es ist kalt hier drinnen.
> der Schlüssel	the key	Ich habe meinen Schlüssel schon wieder zu Hause vergessen.
> anrufen	to call	Ich rufe dich morgen nach der Arbeit an, versprochen.
> ```
>
> Here are the words:
>
> ```
> PASTE YOUR WORD LIST HERE, ONE PER LINE
> ```

---

## Prompt B — generate the words as well

Same as Prompt A, but replace the final section with:

> Generate 50 German words for a learner at CEFR level **A2–B1**, on the
> theme of **[YOUR TOPIC — e.g. the kitchen, travel and transport, work and
> office, feelings and opinions]**. Pick words that are genuinely common in
> everyday use, not textbook curiosities. Mix nouns, verbs, and adjectives
> rather than giving only nouns.

---

## If the tabs come out wrong

Copying out of a chat window occasionally converts tabs to spaces, which
silently breaks the columns. If that happens, ask for pipe separators
instead — the loader accepts either, detecting which from the header row:

> Use ` | ` (space, pipe, space) between columns instead of tabs. Everything
> else stays the same.

```
german | english | phrase
das Fenster | the window | Bitte mach das Fenster zu, es ist kalt hier drinnen.
```

---

## Checking the file before you run the build

`python -m germanaudio vocab.tsv --check` runs validation only, without
synthesizing anything. It reports every malformed line at once with its line
number, so a batch that came out wrong is a quick fix rather than a hunt.
