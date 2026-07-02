"""Peak VRAM measurement for StyleTTS2 during a single generation call."""

import torch
from styletts2.tts import StyleTTS2

TEXT = (
    "Great, I love that question! So we know that sunlight is important for "
    "photosynthesis, but let's think a little deeper."
)


def main():
    torch.cuda.reset_peak_memory_stats()
    model = StyleTTS2()
    n_params = sum(p.numel() for p in model.model.decoder.parameters()) + sum(
        p.numel() for p in model.model.predictor.parameters()
    )

    torch.cuda.reset_peak_memory_stats()
    model.inference(TEXT)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"model=styletts2 params={n_params/1e6:.1f}M(partial) peak_vram_gb={peak_gb:.3f}")


if __name__ == "__main__":
    main()
