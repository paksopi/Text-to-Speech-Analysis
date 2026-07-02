"""Peak VRAM measurement for MMS-TTS (Malay checkpoint) during a single generation call."""

import sys
from pathlib import Path

import torch
from transformers import VitsModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_data import MENTOR_SENTENCES  # noqa: E402

TEXT = MENTOR_SENTENCES["bm"]
MODEL_ID = "facebook/mms-tts-zlm"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats()

    model = VitsModel.from_pretrained(MODEL_ID).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    n_params = sum(p.numel() for p in model.parameters())

    inputs = tokenizer(TEXT, return_tensors="pt").to(device)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(**inputs)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"model=mms_tts params={n_params / 1e6:.1f}M peak_vram_gb={peak_gb:.3f}")


if __name__ == "__main__":
    main()
