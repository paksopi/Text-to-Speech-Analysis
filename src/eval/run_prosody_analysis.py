"""Prosody-delta analysis: does the control instruction measurably change VoxCPM2's
output, or is it cosmetic? Compares en_baseline.wav (with the 'calm, articulate,
reflective...' control) against en_uncontrolled.wav (same text, no control), both
generated from the same model on the same text.

Metrics:
- F0 (pitch) mean/std, via librosa.pyin -- a flatter, lower-variance F0 std is
  consistent with a "calm, reflective" delivery; a wider F0 std suggests more
  animated/expressive delivery.
- Speaking rate (words per second of *voiced* audio).
- Pause ratio (fraction of total duration that is silence/pause), via
  librosa.effects.split -- "deliberate pauses... to give the listener space to
  think" should show up as a higher pause ratio.
"""

import json
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "results" / "audio"
OUT_PATH = ROOT / "results" / "eval" / "prosody_results.json"

TEXT_WORD_COUNT = len(
    (
        "Great, I love that question! So we know that sunlight is important for "
        "photosynthesis, but let's think a little deeper. Think of a plant as a little "
        "factory, light is the energy that comes in. Now, what do you think a plant does "
        "with that energy? What do plants need to survive and grow?"
    ).split()
)


def analyze(path: Path) -> dict:
    y, sr = librosa.load(path, sr=None)
    duration = len(y) / sr

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    f0_voiced = f0[voiced_flag]

    intervals = librosa.effects.split(y, top_db=30)
    voiced_duration = sum((end - start) for start, end in intervals) / sr
    pause_duration = duration - voiced_duration
    pause_ratio = pause_duration / duration

    speaking_rate = TEXT_WORD_COUNT / voiced_duration if voiced_duration > 0 else 0.0

    return {
        "file": path.name,
        "duration_s": round(duration, 3),
        "f0_mean_hz": round(float(np.nanmean(f0_voiced)), 2) if len(f0_voiced) else None,
        "f0_std_hz": round(float(np.nanstd(f0_voiced)), 2) if len(f0_voiced) else None,
        "pause_ratio": round(pause_ratio, 4),
        "speaking_rate_wps": round(speaking_rate, 3),
        "n_pauses": len(intervals) - 1,
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    controlled = analyze(AUDIO_DIR / "en_baseline.wav")
    uncontrolled = analyze(AUDIO_DIR / "en_uncontrolled.wav")

    results = {
        "controlled (calm/articulate/reflective instruction)": controlled,
        "uncontrolled (no control instruction)": uncontrolled,
    }
    print(json.dumps(results, indent=2))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] wrote results to {OUT_PATH}")


if __name__ == "__main__":
    main()
