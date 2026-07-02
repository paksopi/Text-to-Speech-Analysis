"""StyleTTS2 smoke test: EN only (English-only phonemizer, no BM/ID support)."""

import time
from pathlib import Path

from styletts2.tts import StyleTTS2

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "results" / "comparison"
LOG_PATH = OUT_DIR / "styletts2_timings.tsv"

TEXT_EN = (
    "Great, I love that question! So we know that sunlight is important for "
    "photosynthesis, but let's think a little deeper."
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = StyleTTS2()

    out_path = OUT_DIR / "styletts2_en.wav"
    start = time.perf_counter()
    model.inference(TEXT_EN, output_wav_file=str(out_path))
    elapsed = time.perf_counter() - start

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"styletts2_en.wav\t{elapsed:.3f}s\t{TEXT_EN!r}\n")
    print(f"[done] styletts2_en.wav in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
