"""MMS-TTS smoke test: Malay (facebook/mms-tts-zlm) + Indonesian (facebook/mms-tts-ind).

Not part of the original 10-model catalog; added per the README's paused
in-progress notes as a lightweight VITS-based model that actually claims
Malay/Indonesian support. License: CC-BY-NC 4.0 (non-commercial).
"""

import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import VitsModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_data import SHORT_SENTENCES, MENTOR_SENTENCES  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "comparison"

MODELS = {
    "ms": "facebook/mms-tts-zlm",
    "id": "facebook/mms-tts-ind",
}

TEXTS = {
    "ms": MENTOR_SENTENCES["bm"],
    "id": SHORT_SENTENCES["id"],
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for lang, model_id in MODELS.items():
        text = TEXTS[lang]
        model = VitsModel.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        start = time.perf_counter()
        with torch.no_grad():
            output = model(**inputs).waveform
        elapsed = time.perf_counter() - start

        out_path = OUT_DIR / f"mms_tts_{lang}.wav"
        sf.write(str(out_path), output.squeeze().cpu().numpy(), model.config.sampling_rate)
        print(f"[done] mms_tts_{lang}.wav in {elapsed:.3f}s -> {out_path}")


if __name__ == "__main__":
    main()
