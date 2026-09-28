"""Synthesize a handful of samples so a human can judge German quality.

Run this before building anything else. If the German is wrong, the engine
gets swapped before the rest of the pipeline is written.

    python scripts/listening_test.py
"""

from pathlib import Path

import numpy as np
from supertonic import TTS

OUT = Path("out/listening_test")

SAMPLES = [
    ("de", "das Fenster"),
    ("de", "der Schlüssel"),
    ("de", "die Möglichkeit"),
    ("de", "Bitte mach das Fenster zu, es ist kalt hier drinnen."),
    ("de", "Ich habe meinen Schlüssel schon wieder zu Hause vergessen."),
    ("de", "Er hat sich über die schlechte Nachricht sehr geärgert."),
    ("en", "the window"),
    ("en", "the key"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    tts = TTS(auto_download=True)
    style = tts.get_voice_style(voice_name="M1")

    for index, (lang, text) in enumerate(SAMPLES, start=1):
        wav, duration = tts.synthesize(text, voice_style=style, lang=lang)

        if index == 1:
            # Record what the engine actually hands back — Task 4's adapter
            # has to normalize this to float32 mono.
            array = np.asarray(wav)
            print("--- engine facts (note these down) ---")
            print(f"  type returned : {type(wav)}")
            print(f"  dtype         : {array.dtype}")
            print(f"  shape         : {array.shape}")
            print(f"  min / max     : {array.min():.4f} / {array.max():.4f}")
            print(f"  duration (s)  : {duration}")
            print("--------------------------------------")

        path = OUT / f"{index:02d}_{lang}.wav"
        tts.save_audio(wav, str(path))
        # synthesize() returns duration as a 1-element ndarray, not a scalar.
        print(f"{path}  ({float(duration[0]):.2f}s)  {text}")


if __name__ == "__main__":
    main()
