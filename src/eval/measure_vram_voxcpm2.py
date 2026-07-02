"""Peak VRAM measurement for VoxCPM2 during a single generation call."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import controlled_text, load_model  # noqa: E402
from text_data import SHORT_SENTENCES  # noqa: E402

TEXT = SHORT_SENTENCES["en"]


def main():
    torch.cuda.reset_peak_memory_stats()
    model, _ = load_model(optimize=False, load_denoiser=False)

    n_params = sum(p.numel() for p in model.tts_model.parameters())
    torch.cuda.reset_peak_memory_stats()
    model.generate(text=controlled_text(TEXT))

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"model=voxcpm2 params={n_params/1e6:.1f}M peak_vram_gb={peak_gb:.3f}")


if __name__ == "__main__":
    main()
