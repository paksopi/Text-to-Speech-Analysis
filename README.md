# TTS Model Analysis — Malay / Indonesian / English

An analysis and hands-on comparison of open-source text-to-speech (TTS) models for Malay (BM),
Indonesian (ID), and English (EN) speech synthesis with zero-shot voice cloning. 10 models were
researched; all 10 were actually installed and run (or documented as blocked) on the same hardware.
**[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)** (Apache-2.0, open-weight) is the model selected
for this project — see [Why VoxCPM2](#why-voxcpm2) below — and the rest of this repo builds out its PoC
in more depth: environment setup, model download, EN/BM/ID generation tests, and zero-shot voice cloning.

- Full model comparison, measured results, and install-friction notes: [`reports/tts_models_comparison.md`](reports/tts_models_comparison.md)
- VoxCPM2 infrastructure/feasibility writeup: [`reports/VoxCPM2_PoC_Infrastructure_Proposal.md`](reports/VoxCPM2_PoC_Infrastructure_Proposal.md)
- VoxCPM2 PoC rerun results (after the original was lost and rebuilt): [`reports/poc_rerun_results.md`](reports/poc_rerun_results.md)

## Why VoxCPM2

10 models were compared: VoxCPM2, Piper, Coqui XTTSv2, StyleTTS2, Parler-TTS Mini, ChatTTS, F5-TTS,
MeloTTS, Fish Speech, and CosyVoice 2.0. **VoxCPM2 is the only one with native Malay support**, and one
of only two (with Piper) supporting Indonesian — every other model tops out at English, or
English+Chinese. 3 of the 9 alternatives (MeloTTS, Fish Speech, CosyVoice) couldn't even be installed
cleanly (broken/incomplete PyPI packages, or no official package at all). This settles the model choice
for this project regardless of speed/VRAM trade-offs — full measured timings and install notes in the
comparison report.

| Model | EN | BM | ID | Measured speed | Device | Status |
|---|---|---|---|---|---|---|
| **VoxCPM2** (chosen) | ✅ | ✅ | ✅ | 20.45s / 18.97s / 29.13s | GPU | Full native support |
| Piper | ✅ | ❌ | ✅ | 0.384s / — / 0.307s | CPU | Only other model with any ID support |
| Coqui XTTSv2 | ✅ | ❌ | ❌ | 4.79s / — / — | GPU | 17 languages, none BM/ID |
| StyleTTS2 | ✅ | ❌ | ❌ | 3.86s / — / — | GPU | English-only |
| Parler-TTS Mini | ✅ | ❌ | ❌ | 16.575s / — / — | GPU | English-only |
| ChatTTS | ✅ | ❌ | ❌ | 13.107s / — / — | GPU | EN+ZH only |
| F5-TTS | ✅ | ❌ | ❌ | 15.123s / — / — | GPU | EN+ZH only |
| MeloTTS | — | ❌ | ❌ | — | — | Blocked (broken PyPI package) |
| Fish Speech | — | ❌ | ❌ | — | — | Blocked (PyPI package inference-incomplete) |
| CosyVoice 2.0 | — | ❌ | ❌ | — | — | Skipped (no official PyPI package) |

## VoxCPM2 PoC Results

Rebuilt and reran on the same RTX 3050 Laptop (6GB VRAM) hardware as the original proposal, across two
runs. All twelve test generations succeeded — EN/BM/ID baseline (default voice) and EN/BM/ID cloned from
a single anchor clip (the EN baseline output, reused as the zero-shot voice-cloning reference across all
three languages) — see the rerun report for the full test script/control-instruction text.

**Run 1 (eager mode — `torch.compile` silently disabled by a Triton version mismatch):**

| Language | Baseline time | Cloned time |
|---|---|---|
| EN | 20.45s | 56.05s |
| BM | 18.97s | 45.59s |
| ID | 29.13s | 49.49s |

**Run 2 (`torch.compile` fixed by pinning `triton-windows==3.2.0.post21`):**

| Language | Baseline time | Cloned time |
|---|---|---|
| EN | 269.76s | 236.15s |
| BM | 353.78s | 348.81s |
| ID | 359.44s | 328.52s |

`torch.compile` now genuinely activates, but is **10-20x slower** here than eager mode — it recompiles a
fresh graph for every distinct input shape, and every call in this run used a different one, so
compilation cost never amortized. This isn't a contradiction of the original proposal's RTX 4090 compile
projections, which assume a production serving loop with many repeated, shape-stable calls; a handful of
one-off, varying-length test calls is close to the worst case for `torch.compile`. Practical takeaway:
keep `optimize=False` (eager mode) for exploratory testing like this, and only enable compile once
building a real serving loop. Full breakdown and open items in the rerun report.

## Layout

| Folder | Contents |
|---|---|
| `src/` | VoxCPM2 generation scripts (baseline TTS + voice-cloning tests) |
| `src/comparison/` | One-off smoke-test scripts for each of the 9 alternative models |
| `results/audio/` | VoxCPM2 `.wav` test outputs (baseline and cloned-voice, per language) |
| `results/comparison/` | `.wav` outputs and timing logs from the 9-model comparison |
| `results/logs/` | Per-call generation timing logs for VoxCPM2 |
| `reports/` | Written analysis — the model comparison, the original infra proposal, and the PoC rerun writeup |

## Setup

Requires **Python 3.10–3.12**. GPU: NVIDIA, CUDA 12.0+ driver, 8GB+ VRAM recommended (this PoC was
run on a 6GB card — see the report for the tradeoffs that implies).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Each model in `src/comparison/` was tested in its own isolated venv (`.venvs/<model>/`, gitignored) to
avoid dependency conflicts between frameworks — there's no single shared `requirements.txt` for those;
see the comparison report for each model's install notes.

## Usage

```bash
python src/generate_baseline.py    # VoxCPM2, EN/BM/ID, default voice
python src/generate_cloned.py      # VoxCPM2, EN/BM/ID, cloned from the EN baseline as the anchor voice
```

```bash
python src/comparison/run_piper.py         # Piper (EN/ID)
python src/comparison/run_coqui_xtts.py    # Coqui XTTSv2 (EN, default + cloned)
python src/comparison/run_styletts2.py     # StyleTTS2 (EN)
python src/comparison/run_parler.py        # Parler-TTS Mini (EN)
python src/comparison/run_chattts.py       # ChatTTS (EN)
python src/comparison/run_f5tts.py         # F5-TTS (EN, cloned)
```

## References

- [VoxCPM2 (Hugging Face)](https://huggingface.co/openbmb/VoxCPM2)
- [ElevenLabs](https://elevenlabs.io/) — commercial TTS API benchmarked against in the proposal
