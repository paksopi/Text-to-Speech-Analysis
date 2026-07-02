"""Parler-TTS Mini smoke test: EN only (model card lists English-only training data)."""

import time
from pathlib import Path

import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "results" / "comparison"
LOG_PATH = OUT_DIR / "parler_timings.tsv"

TEXT_EN = (
    "Great, I love that question! So we know that sunlight is important for "
    "photosynthesis, but let's think a little deeper."
)
DESCRIPTION = (
    "A calm, articulate female speaker delivers her speech at a moderate pace "
    "with clear audio quality."
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    input_ids = tokenizer(DESCRIPTION, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(TEXT_EN, return_tensors="pt").input_ids.to(device)

    start = time.perf_counter()
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
    elapsed = time.perf_counter() - start

    audio = generation.cpu().numpy().squeeze()
    out_path = OUT_DIR / "parler_en.wav"
    sf.write(out_path, audio, model.config.sampling_rate)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"parler_en.wav\t{elapsed:.3f}s\t{device}\t{TEXT_EN!r}\n")
    print(f"[done] parler_en.wav in {elapsed:.3f}s -> {out_path}")


if __name__ == "__main__":
    main()
