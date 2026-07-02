"""Peak VRAM measurement for F5-TTS during a single generation call."""

from pathlib import Path

import torch
from f5_tts.api import F5TTS

ROOT = Path(__file__).resolve().parent.parent.parent
ANCHOR = ROOT / "results" / "audio" / "en_baseline.wav"

REF_TEXT = (
    "Every skill you build today becomes the foundation for tomorrow's confidence, "
    "so keep practicing with patience and curiosity."
)
GEN_TEXT = (
    "Great, I love that question! So we know that sunlight is important for "
    "photosynthesis, but let's think a little deeper."
)


def main():
    torch.zeros(1).cuda()  # ensure CUDA context is initialized before any reset
    f5tts = F5TTS()

    out_path = ROOT / "results" / "eval" / "_f5tts_vram_probe.wav"
    torch.cuda.reset_peak_memory_stats()
    f5tts.infer(ref_file=str(ANCHOR), ref_text=REF_TEXT, gen_text=GEN_TEXT, file_wave=str(out_path))

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"model=f5tts peak_vram_gb={peak_gb:.3f}")


if __name__ == "__main__":
    main()
