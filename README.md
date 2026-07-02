# VoxCPM2 TTS PoC — Malay / Indonesian / English

A proof-of-concept using [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (Apache-2.0, open-weight),
an emotion-controllable text-to-speech model, for Malay (BM), Indonesian (ID), and English (EN) speech
synthesis with zero-shot voice cloning.

Full infrastructure/feasibility writeup: [`reports/VoxCPM2_PoC_Infrastructure_Proposal.md`](reports/VoxCPM2_PoC_Infrastructure_Proposal.md).

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
