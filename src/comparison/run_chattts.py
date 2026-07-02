"""ChatTTS smoke test: EN + ZH only (no BM/ID support per model docs)."""

import time
from pathlib import Path

import ChatTTS
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "results" / "comparison"
LOG_PATH = OUT_DIR / "chattts_timings.tsv"

TEXT_EN = (
    "Great, I love that question! So we know that sunlight is important for "
    "photosynthesis, but let's think a little deeper."
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chat = ChatTTS.Chat()
    chat.load(source="huggingface", device=device)

    start = time.perf_counter()
    wavs = chat.infer([TEXT_EN])
    elapsed = time.perf_counter() - start

    out_path = OUT_DIR / "chattts_en.wav"
    sf.write(out_path, wavs[0].squeeze(), 24000)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"chattts_en.wav\t{elapsed:.3f}s\t{device}\t{TEXT_EN!r}\n")
    print(f"[done] chattts_en.wav in {elapsed:.3f}s -> {out_path}")


if __name__ == "__main__":
    main()
