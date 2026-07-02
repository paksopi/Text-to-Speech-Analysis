"""Kokoro-82M smoke test: EN only (no Malay/Indonesian support).

Not part of the original 10-model catalog; added per the README's paused
in-progress notes as a very lightweight (82M param) Apache-2.0 model, run
for a broader personal report despite not meeting this project's BM/ID
requirement.
"""

import sys
import time
from pathlib import Path

import soundfile as sf
from kokoro import KPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_data import SHORT_SENTENCES  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "comparison"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = KPipeline(lang_code="a")  # 'a' = American English

    start = time.perf_counter()
    generator = pipeline(SHORT_SENTENCES["en"], voice="af_heart")
    chunks = [audio for _, _, audio in generator]
    elapsed = time.perf_counter() - start

    import numpy as np

    audio = np.concatenate(chunks)
    out_path = OUT_DIR / "kokoro_en.wav"
    sf.write(str(out_path), audio, 24000)
    print(f"[done] kokoro_en.wav in {elapsed:.3f}s -> {out_path}")


if __name__ == "__main__":
    main()
