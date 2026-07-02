"""Peak VRAM measurement for Coqui XTTSv2 during a single generation call."""

import sys
from pathlib import Path

import torch
from TTS.api import TTS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_data import SHORT_SENTENCES  # noqa: E402

TEXT = SHORT_SENTENCES["en"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    n_params = sum(p.numel() for p in tts.synthesizer.tts_model.parameters())

    torch.cuda.reset_peak_memory_stats()
    tts.tts(text=TEXT, speaker=tts.speakers[0] if tts.speakers else None, language="en")

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"model=coqui_xtts params={n_params / 1e6:.1f}M peak_vram_gb={peak_gb:.3f}")


if __name__ == "__main__":
    main()
