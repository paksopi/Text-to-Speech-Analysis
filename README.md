# VoxCPM2 TTS PoC — Malay / Indonesian / English

A proof-of-concept using [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (Apache-2.0, open-weight),
an emotion-controllable text-to-speech model, for Malay (BM), Indonesian (ID), and English (EN) speech
synthesis with zero-shot voice cloning.

Full infrastructure/feasibility writeup: [`reports/VoxCPM2_PoC_Infrastructure_Proposal.md`](reports/VoxCPM2_PoC_Infrastructure_Proposal.md).
Rerun results after the original PoC was lost and rebuilt: [`reports/poc_rerun_results.md`](reports/poc_rerun_results.md).

## Results

Rebuilt and reran on the same RTX 3050 Laptop (6GB VRAM) hardware as the original proposal. All six test
generations succeeded — EN/BM/ID baseline (default voice) and EN/BM/ID cloned from a single anchor clip
(the EN baseline output, reused as the zero-shot voice-cloning reference across all three languages).

| Language | Baseline time | Cloned time |
|---|---|---|
| EN | 20.45s | 56.05s |
| BM | 18.97s | 45.59s |
| ID | 29.13s | 49.49s |

`torch.compile`/Triton was installed and requested but VoxCPM2 disabled it internally at load time due
to a Triton API mismatch (`AttrsDescriptor` import failure) — generation ran in eager mode. See the rerun
report for the full breakdown and open items.

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
