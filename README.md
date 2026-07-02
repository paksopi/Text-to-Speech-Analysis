# VoxCPM2 TTS PoC — Malay / Indonesian / English

A proof-of-concept using [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (Apache-2.0, open-weight),
an emotion-controllable text-to-speech model, for Malay (BM), Indonesian (ID), and English (EN) speech
synthesis with zero-shot voice cloning.

Full infrastructure/feasibility writeup: [`reports/VoxCPM2_PoC_Infrastructure_Proposal.md`](reports/VoxCPM2_PoC_Infrastructure_Proposal.md).
Rerun results after the original PoC was lost and rebuilt: [`reports/poc_rerun_results.md`](reports/poc_rerun_results.md).
Measured comparison against 9 alternative open-source TTS models: [`reports/tts_models_comparison.md`](reports/tts_models_comparison.md).

## Why VoxCPM2

9 alternative open-source TTS models (Piper, Coqui XTTSv2, StyleTTS2, Parler-TTS Mini, ChatTTS, F5-TTS,
MeloTTS, Fish Speech, CosyVoice 2.0) were installed and tested on the same hardware for comparison.
**VoxCPM2 is the only one with native Malay support**, and one of only two (with Piper) supporting
Indonesian — every other model tops out at English, or English+Chinese. 3 of the 9 alternatives (MeloTTS,
Fish Speech, CosyVoice) couldn't even be installed cleanly (broken/incomplete PyPI packages, or no
official package at all). Full measured timings and install notes in the comparison report above.

## Results

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
| `src/` | Generation scripts (baseline TTS + voice-cloning tests) |
| `results/audio/` | Generated `.wav` test outputs (baseline and cloned-voice, per language) |
| `results/logs/` | Per-call generation timing logs |
| `reports/` | Written analysis — the original infra proposal, plus the PoC rerun writeup |

## Setup

Requires **Python 3.10–3.12**. GPU: NVIDIA, CUDA 12.0+ driver, 8GB+ VRAM recommended (this PoC was
run on a 6GB card — see the report for the tradeoffs that implies).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Usage

```bash
python src/generate_baseline.py    # EN/BM/ID, default voice
python src/generate_cloned.py      # EN/BM/ID, cloned from the EN baseline as the anchor voice
```

## References

- [VoxCPM2 (Hugging Face)](https://huggingface.co/openbmb/VoxCPM2)
- [ElevenLabs](https://elevenlabs.io/) — commercial TTS API benchmarked against in the proposal
