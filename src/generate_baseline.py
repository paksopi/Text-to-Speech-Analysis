"""Generate baseline EN/BM/ID test utterances with the default (uncloned) voice."""

from common import AUDIO_DIR, LOG_DIR, TEST_SENTENCES, controlled_text, generate_and_save, load_model


def main():
    model, optimize_used = load_model(optimize=True, load_denoiser=False)
    print(f"[info] model loaded (torch.compile optimize={optimize_used})")

    log_path = LOG_DIR / "baseline_timings.tsv"
    for lang, text in TEST_SENTENCES.items():
        out_path = AUDIO_DIR / f"{lang}_baseline.wav"
        generate_and_save(model, controlled_text(text), out_path, log_path)


if __name__ == "__main__":
    main()
