"""Shared text constants used across generation, comparison, and evaluation scripts.

Before this module existed, the same sentences were hand-copied into a dozen
files (src/common.py, src/eval/manifest.py, every src/comparison/run_*.py, and
every src/eval/measure_vram_*.py). That made it easy for a copy to drift from
the others without anyone noticing. Everything text-related now lives here
once; scripts import from this module instead of redefining it.
"""

# Long "mentor" test sentences used for the VoxCPM2 PoC (baseline + cloned
# generations, all three languages). These are also the expected-text values
# for the WER evaluation of the VoxCPM2 outputs.
MENTOR_SENTENCES = {
    "en": (
        "Great, I love that question! So we know that sunlight is important for "
        "photosynthesis — but let's think a little deeper. Think of a plant as a little "
        "factory — light is the energy that comes in. Now, what do you think a plant does "
        "with that energy? What do plants need to survive and grow?"
    ),
    "bm": (
        "Bagus, saya suka soalan itu! Jadi kita tahu bahawa cahaya matahari penting "
        "untuk fotosintesis — tetapi mari kita fikir dengan lebih mendalam. Bayangkan "
        "tumbuhan sebagai sebuah kilang kecil — cahaya ialah tenaga yang masuk. Sekarang, "
        "apa pula yang anda fikir tumbuhan lakukan dengan tenaga itu? Apakah yang tumbuhan "
        "perlukan untuk hidup dan membesar?"
    ),
    "id": (
        "Bagus, aku suka pertanyaan itu! Jadi kita tahu bahwa sinar matahari penting "
        "untuk fotosintesis — tapi mari kita berpikir lebih dalam. Bayangkan tumbuhan "
        "sebagai sebuah pabrik kecil — cahaya adalah energi yang masuk. Sekarang, menurutmu "
        "apa yang dilakukan tumbuhan dengan energi itu? Apa yang dibutuhkan tumbuhan untuk "
        "bertahan hidup dan tumbuh?"
    ),
}

# Shorter sentences used for the one-off comparison smoke tests
# (src/comparison/run_*.py) and the VRAM/param-count probes
# (src/eval/measure_vram_*.py) -- those models were only smoke-tested on a
# truncated sentence, not the full mentor paragraph above.
SHORT_SENTENCES = {
    "en": (
        "Great, I love that question! So we know that sunlight is important for "
        "photosynthesis, but let's think a little deeper."
    ),
    "id": (
        "Bagus, aku suka pertanyaan itu! Jadi kita tahu bahwa sinar matahari penting "
        "untuk fotosintesis, tapi mari kita berpikir lebih dalam."
    ),
}

# Reference-clip transcript for F5-TTS. F5-TTS is a flow-matching model with
# no built-in default voice, so every call requires a reference clip plus its
# exact transcript; this project reuses the VoxCPM2 EN baseline as that clip.
F5TTS_REF_TEXT = (
    "Every skill you build today becomes the foundation for tomorrow's confidence, "
    "so keep practicing with patience and curiosity."
)

# Natural-language voice description fed to Parler-TTS Mini in place of a
# reference clip (Parler conditions generation on a text description of the
# voice, not on audio).
PARLER_DESCRIPTION = (
    "A calm, articulate female speaker delivers her speech at a moderate pace with clear audio quality."
)
