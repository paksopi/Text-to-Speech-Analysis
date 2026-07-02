"""Peak VRAM measurement for Parler-TTS Mini during a single generation call."""

import sys
from pathlib import Path

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_data import PARLER_DESCRIPTION, SHORT_SENTENCES  # noqa: E402

TEXT_EN = SHORT_SENTENCES["en"]
DESCRIPTION = PARLER_DESCRIPTION


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats()

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    n_params = sum(p.numel() for p in model.parameters())

    input_ids = tokenizer(DESCRIPTION, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(TEXT_EN, return_tensors="pt").input_ids.to(device)

    torch.cuda.reset_peak_memory_stats()
    model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"model=parler params={n_params/1e6:.1f}M peak_vram_gb={peak_gb:.3f}")


if __name__ == "__main__":
    main()
