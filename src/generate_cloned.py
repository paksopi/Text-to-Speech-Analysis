"""Generate EN/BM/ID test utterances cloned from the EN baseline as the anchor voice.

Run generate_baseline.py first — this reuses results/audio/en_baseline.wav as the
zero-shot voice-cloning reference clip, matching the proposal's "one reference clip
reused across all 3 languages" validation.
"""

from common import AUDIO_DIR, LOG_DIR, TEST_SENTENCES, controlled_text, generate_and_save, load_model


def main():
    anchor = AUDIO_DIR / "en_baseline.wav"
    if not anchor.exists():
        raise FileNotFoundError(f"Anchor clip not found: {anchor} — run generate_baseline.py first")

    model, optimize_used = load_model(optimize=True, load_denoiser=False)
    print(f"[info] model loaded (torch.compile optimize={optimize_used})")
    print(f"[info] using anchor clip: {anchor}")

    log_path = LOG_DIR / "cloned_timings.tsv"
    for lang, text in TEST_SENTENCES.items():
        out_path = AUDIO_DIR / f"{lang}_cloned.wav"
        generate_and_save(
            model,
            controlled_text(text),
            out_path,
            log_path,
            reference_wav_path=str(anchor),
        )


if __name__ == "__main__":
    main()
