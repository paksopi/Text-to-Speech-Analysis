"""Piper TTS smoke test: EN + ID (no Malay voice available in rhasspy/piper-voices)."""

import sys
import time
import wave
from pathlib import Path

from piper import PiperVoice

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_data import SHORT_SENTENCES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
VOICE_DIR = ROOT / ".cache" / "piper_voices"
OUT_DIR = ROOT / "results" / "comparison"
LOG_PATH = OUT_DIR / "piper_timings.tsv"

TESTS = {
    "en": ("en_US-amy-medium.onnx", SHORT_SENTENCES["en"]),
    "id": ("id_ID-news_tts-medium.onnx", SHORT_SENTENCES["id"]),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for lang, (voice_file, text) in TESTS.items():
        voice = PiperVoice.load(str(VOICE_DIR / voice_file))
        out_path = OUT_DIR / f"piper_{lang}.wav"

        start = time.perf_counter()
        with wave.open(str(out_path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        elapsed = time.perf_counter() - start

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"piper_{lang}.wav\t{elapsed:.3f}s\tCPU\t{text!r}\n")
        print(f"[done] piper_{lang}.wav in {elapsed:.3f}s -> {out_path}")


if __name__ == "__main__":
    main()
