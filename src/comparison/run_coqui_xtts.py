"""Coqui XTTSv2 smoke test: EN only (BM/ID not among XTTSv2's 17 supported languages).

Runs a plain EN generation and an EN voice-clone using the VoxCPM2 en_baseline.wav
as the reference clip, for an apples-to-apples anchor comparison.
"""

import time
from pathlib import Path

import torch
from TTS.api import TTS

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "results" / "comparison"
LOG_PATH = OUT_DIR / "coqui_xtts_timings.tsv"
ANCHOR = ROOT / "results" / "audio" / "en_baseline.wav"

TEXT_EN = (
    "Great, I love that question! So we know that sunlight is important for "
    "photosynthesis, but let's think a little deeper."
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # Default speaker (built-in, no reference clip)
    start = time.perf_counter()
    tts.tts_to_file(
        text=TEXT_EN,
        speaker=tts.speakers[0] if tts.speakers else None,
        language="en",
        file_path=str(OUT_DIR / "coqui_xtts_en.wav"),
    )
    elapsed = time.perf_counter() - start
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"coqui_xtts_en.wav\t{elapsed:.3f}s\t{device}\tdefault-speaker\n")
    print(f"[done] coqui_xtts_en.wav in {elapsed:.3f}s")

    # Cloned from the VoxCPM2 EN anchor clip
    start = time.perf_counter()
    tts.tts_to_file(
        text=TEXT_EN,
        speaker_wav=str(ANCHOR),
        language="en",
        file_path=str(OUT_DIR / "coqui_xtts_en_cloned.wav"),
    )
    elapsed = time.perf_counter() - start
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"coqui_xtts_en_cloned.wav\t{elapsed:.3f}s\t{device}\tcloned-from-anchor\n")
    print(f"[done] coqui_xtts_en_cloned.wav in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
