"""Peak VRAM measurement for ChatTTS during a single generation call."""

import sys
from pathlib import Path

import torch
import ChatTTS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_data import SHORT_SENTENCES  # noqa: E402

TEXT = SHORT_SENTENCES["en"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats()

    chat = ChatTTS.Chat()
    chat.load(source="huggingface", device=device)

    torch.cuda.reset_peak_memory_stats()
    chat.infer([TEXT])

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"model=chattts peak_vram_gb={peak_gb:.3f}")


if __name__ == "__main__":
    main()
