"""Manifest of every generated wav in this repo: path, language, expected text, model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_data import MENTOR_SENTENCES, SHORT_SENTENCES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "results" / "audio"
COMPARISON_DIR = ROOT / "results" / "comparison"

TEXT_EN_SHORT = SHORT_SENTENCES["en"]
TEXT_ID_SHORT = SHORT_SENTENCES["id"]

TEXT_EN_MENTOR = MENTOR_SENTENCES["en"]
TEXT_BM_MENTOR = MENTOR_SENTENCES["bm"]
TEXT_ID_MENTOR = MENTOR_SENTENCES["id"]

# (path, model, lang_code, expected_text, is_cloned, anchor_path_or_None)
MANIFEST = [
    (AUDIO_DIR / "en_baseline.wav", "voxcpm2", "en", TEXT_EN_MENTOR, False, None),
    (AUDIO_DIR / "bm_baseline.wav", "voxcpm2", "ms", TEXT_BM_MENTOR, False, None),
    (AUDIO_DIR / "id_baseline.wav", "voxcpm2", "id", TEXT_ID_MENTOR, False, None),
    (AUDIO_DIR / "en_cloned.wav", "voxcpm2", "en", TEXT_EN_MENTOR, True, AUDIO_DIR / "en_baseline.wav"),
    (AUDIO_DIR / "bm_cloned.wav", "voxcpm2", "ms", TEXT_BM_MENTOR, True, AUDIO_DIR / "en_baseline.wav"),
    (AUDIO_DIR / "id_cloned.wav", "voxcpm2", "id", TEXT_ID_MENTOR, True, AUDIO_DIR / "en_baseline.wav"),
    (COMPARISON_DIR / "piper_en.wav", "piper", "en", TEXT_EN_SHORT, False, None),
    (COMPARISON_DIR / "piper_id.wav", "piper", "id", TEXT_ID_SHORT, False, None),
    (COMPARISON_DIR / "coqui_xtts_en.wav", "coqui_xtts", "en", TEXT_EN_SHORT, False, None),
    (
        COMPARISON_DIR / "coqui_xtts_en_cloned.wav",
        "coqui_xtts",
        "en",
        TEXT_EN_SHORT,
        True,
        AUDIO_DIR / "en_baseline.wav",
    ),
    (COMPARISON_DIR / "styletts2_en.wav", "styletts2", "en", TEXT_EN_SHORT, False, None),
    (COMPARISON_DIR / "parler_en.wav", "parler", "en", TEXT_EN_SHORT, False, None),
    (COMPARISON_DIR / "chattts_en.wav", "chattts", "en", TEXT_EN_SHORT, False, None),
    (
        COMPARISON_DIR / "f5tts_en.wav",
        "f5tts",
        "en",
        TEXT_EN_SHORT,
        True,
        AUDIO_DIR / "en_baseline.wav",
    ),
]
