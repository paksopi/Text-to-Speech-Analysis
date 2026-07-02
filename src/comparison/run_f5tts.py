"""F5-TTS smoke test: EN only (officially supports English + Chinese, no BM/ID).

F5-TTS is voice-cloning-only (flow-matching architecture, no built-in default
voice) so it requires a reference clip + its transcript. Reuses the VoxCPM2
EN baseline as the reference, matching the anchor-clip pattern used elsewhere
in this project.
"""

import sys
import time
from pathlib import Path

from f5_tts.api import F5TTS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_data import F5TTS_REF_TEXT, SHORT_SENTENCES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "results" / "comparison"
LOG_PATH = OUT_DIR / "f5tts_timings.tsv"
ANCHOR = ROOT / "results" / "audio" / "en_baseline.wav"

REF_TEXT = F5TTS_REF_TEXT
GEN_TEXT = SHORT_SENTENCES["en"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f5tts = F5TTS()

    out_path = OUT_DIR / "f5tts_en.wav"
    start = time.perf_counter()
    f5tts.infer(
        ref_file=str(ANCHOR),
        ref_text=REF_TEXT,
        gen_text=GEN_TEXT,
        file_wave=str(out_path),
    )
    elapsed = time.perf_counter() - start

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"f5tts_en.wav\t{elapsed:.3f}s\tcloned-from-anchor\t{GEN_TEXT!r}\n")
    print(f"[done] f5tts_en.wav in {elapsed:.3f}s -> {out_path}")


if __name__ == "__main__":
    main()
