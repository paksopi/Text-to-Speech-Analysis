"""Manifest of every generated wav in this repo: path, language, expected text, model."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "results" / "audio"
COMPARISON_DIR = ROOT / "results" / "comparison"

TEXT_EN_SHORT = (
    "Great, I love that question! So we know that sunlight is important for "
    "photosynthesis, but let's think a little deeper."
)
TEXT_ID_SHORT = (
    "Bagus, aku suka pertanyaan itu! Jadi kita tahu bahwa sinar matahari penting "
    "untuk fotosintesis, tapi mari kita berpikir lebih dalam."
)

TEXT_EN_MENTOR = (
    "Great, I love that question! So we know that sunlight is important for "
    "photosynthesis — but let's think a little deeper. Think of a plant as a little "
    "factory — light is the energy that comes in. Now, what do you think a plant does "
    "with that energy? What do plants need to survive and grow?"
)
TEXT_BM_MENTOR = (
    "Bagus, saya suka soalan itu! Jadi kita tahu bahawa cahaya matahari penting "
    "untuk fotosintesis — tetapi mari kita fikir dengan lebih mendalam. Bayangkan "
    "tumbuhan sebagai sebuah kilang kecil — cahaya ialah tenaga yang masuk. Sekarang, "
    "apa pula yang anda fikir tumbuhan lakukan dengan tenaga itu? Apakah yang tumbuhan "
    "perlukan untuk hidup dan membesar?"
)
TEXT_ID_MENTOR = (
    "Bagus, aku suka pertanyaan itu! Jadi kita tahu bahwa sinar matahari penting "
    "untuk fotosintesis — tapi mari kita berpikir lebih dalam. Bayangkan tumbuhan "
    "sebagai sebuah pabrik kecil — cahaya adalah energi yang masuk. Sekarang, menurutmu "
    "apa yang dilakukan tumbuhan dengan energi itu? Apa yang dibutuhkan tumbuhan untuk "
    "bertahan hidup dan tumbuh?"
)

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
