import time
from pathlib import Path

import soundfile as sf

from text_data import MENTOR_SENTENCES

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "results" / "audio"
LOG_DIR = ROOT / "results" / "logs"

CONTROL = (
    "A calm, articulate female voice in her early-20s. Her tone is grounded and "
    "reflective, carrying a sense of 'warm precision'. Deliberate pauses are used "
    "between ideas to give the listener space to think, maintaining a patient, "
    "unhurried pace."
)

# Kept as TEST_SENTENCES here for backwards compatibility with existing
# imports; the actual sentences live in text_data.py so they're defined once.
TEST_SENTENCES = MENTOR_SENTENCES


def controlled_text(text: str, control: str = CONTROL) -> str:
    return f"({control}){text}"


def load_model(optimize: bool = True, load_denoiser: bool = False):
    """Load VoxCPM2, falling back to optimize=False if torch.compile OOMs."""
    from voxcpm import VoxCPM

    try:
        return VoxCPM.from_pretrained(
            "openbmb/VoxCPM2", load_denoiser=load_denoiser, optimize=optimize
        ), optimize
    except Exception as exc:
        if optimize:
            print(f"[warn] optimize=True failed ({exc!r}); retrying with optimize=False")
            return (
                VoxCPM.from_pretrained(
                    "openbmb/VoxCPM2", load_denoiser=load_denoiser, optimize=False
                ),
                False,
            )
        raise


def generate_and_save(model, text: str, out_path: Path, log_path: Path, **kwargs):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    wav = model.generate(text=text, **kwargs)
    elapsed = time.perf_counter() - start

    sf.write(out_path, wav, model.tts_model.sample_rate)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{out_path.name}\t{elapsed:.2f}s\t{text!r}\n")

    print(f"[done] {out_path.name} in {elapsed:.2f}s -> {out_path}")
    return elapsed
