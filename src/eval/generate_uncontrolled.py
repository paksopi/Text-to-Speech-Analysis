"""Generate the EN test sentence with NO control instruction, for prosody comparison
against en_baseline.wav (which uses the 'calm, articulate, reflective' control)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import AUDIO_DIR, LOG_DIR, TEST_SENTENCES, generate_and_save, load_model  # noqa: E402


def main():
    model, optimize_used = load_model(optimize=False, load_denoiser=False)
    out_path = AUDIO_DIR / "en_uncontrolled.wav"
    log_path = LOG_DIR / "uncontrolled_timings.tsv"
    generate_and_save(model, TEST_SENTENCES["en"], out_path, log_path)


if __name__ == "__main__":
    main()
