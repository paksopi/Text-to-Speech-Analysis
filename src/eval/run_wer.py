"""Word Error Rate (WER) evaluation: transcribe every generated wav with Whisper
and compare against the text that was actually requested, per language.

This is the TTS analogue of the language-detection repo's accuracy metric --
it answers "does the model actually say the right words," not just "does it
sound plausible."
"""

import json
from pathlib import Path

import jiwer
import whisper

from manifest import MANIFEST

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_PATH = ROOT / "results" / "eval" / "wer_results.json"

NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model("medium")

    results = []
    for path, model_name, lang, expected_text, is_cloned, anchor in MANIFEST:
        if not path.exists():
            print(f"[skip] {path} not found")
            continue

        transcription = model.transcribe(str(path), language=lang)["text"]
        wer = jiwer.wer(
            expected_text,
            transcription,
            reference_transform=NORMALIZE,
            hypothesis_transform=NORMALIZE,
        )

        result = {
            "file": path.name,
            "model": model_name,
            "lang": lang,
            "cloned": is_cloned,
            "expected_text": expected_text,
            "transcription": transcription.strip(),
            "wer": round(wer, 4),
        }
        results.append(result)
        print(f"[{model_name}/{lang}] {path.name}: WER={wer:.4f}")
        print(f"  expected: {expected_text[:80]}...")
        print(f"  heard:    {transcription.strip()[:80]}...")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wrote {len(results)} results to {OUT_PATH}")


if __name__ == "__main__":
    main()
