"""Peak VRAM measurement for Kokoro-82M during a single generation call."""

import sys
from pathlib import Path

import torch
from kokoro import KPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_data import SHORT_SENTENCES  # noqa: E402

TEXT = SHORT_SENTENCES["en"]


def main():
    torch.cuda.reset_peak_memory_stats()
    pipeline = KPipeline(lang_code="a")
    n_params = sum(p.numel() for p in pipeline.model.parameters())

    torch.cuda.reset_peak_memory_stats()
    generator = pipeline(TEXT, voice="af_heart")
    for _ in generator:
        pass

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"model=kokoro params={n_params / 1e6:.1f}M peak_vram_gb={peak_gb:.3f}")


if __name__ == "__main__":
    main()
