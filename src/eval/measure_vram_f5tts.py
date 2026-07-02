"""Peak VRAM measurement for F5-TTS during a single generation call."""

import sys
from pathlib import Path

import torch
from f5_tts.api import F5TTS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_data import F5TTS_REF_TEXT, SHORT_SENTENCES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
ANCHOR = ROOT / "results" / "audio" / "en_baseline.wav"

REF_TEXT = F5TTS_REF_TEXT
GEN_TEXT = SHORT_SENTENCES["en"]


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
