# TTS Model Analysis — Malay / Indonesian / English

[![CI](https://github.com/paksopi/Text-to-Speech-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/paksopi/Text-to-Speech-Analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

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

| Model | EN | BM | ID | Measured speed | Mode | Device | Status |
|---|---|---|---|---|---|---|---|
| **VoxCPM2** (chosen) | ✅ | ✅ | ✅ | 20.45s / 18.97s / 29.13s | eager | GPU | Full native support |
| Piper | ✅ | ❌ | ✅ | 0.384s / — / 0.307s | default | CPU | Only other model with any ID support |
| Coqui XTTSv2 | ✅ | ❌ | ❌ | 4.79s / — / — | default | GPU | 17 languages, none BM/ID |
| StyleTTS2 | ✅ | ❌ | ❌ | 3.86s / — / — | default | GPU | English-only |
| Parler-TTS Mini | ✅ | ❌ | ❌ | 16.575s / — / — | default | GPU | English-only |
| ChatTTS | ✅ | ❌ | ❌ | 13.107s / — / — | default | GPU | EN+ZH only |
| F5-TTS | ✅ | ❌ | ❌ | 15.123s / — / — | default | GPU | EN+ZH only |
| MeloTTS | — | ❌ | ❌ | — | — | — | Blocked (broken PyPI package) |
| Fish Speech | — | ❌ | ❌ | — | — | — | Blocked (PyPI package inference-incomplete) |
| CosyVoice 2.0 | — | ❌ | ❌ | — | — | — | Skipped (no official PyPI package) |

"Mode" = eager vs. `torch.compile`-compiled inference. VoxCPM2 is the only model here whose wrapper
exposes an explicit eager/compile toggle (`optimize=`); its number above is **eager**, since the PoC rerun
found `torch.compile` is 10-20x *slower* in this one-off-call test pattern — see
[VoxCPM2 PoC Results](#voxcpm2-poc-results) below. "default" for the other 9 models means each library's
own out-of-the-box inference path was used as-is, with no `torch.compile` applied by this project's
scripts — but whether any of those libraries silently self-compile internally wasn't independently
verified, so those numbers should be read as "as installed," not confirmed-eager. **Known limitation:**
this eager/compiled distinction was only tracked for VoxCPM2; it's not recorded for the other 9.

### Objective evaluation, not just language support

Four automated metrics back up the table above with numbers instead of vibes — full methodology and
results in the [comparison report's Objective Evaluation section](reports/tts_models_comparison.md#objective-evaluation).
**Preliminary, not statistically powered:** each number below comes from a single generation per
condition (n=1 per cell, one sentence for the prosody comparison) — useful for spotting large gaps, not
for fine-grained ranking between close scores; see the report's sample-size caveat for detail.

| Metric | What it measures | VoxCPM2 result |
|---|---|---|
| **WER** (Whisper transcription vs. requested text) | Does the model say the right words | **0.0000** across all 6 EN/BM/ID generations — vs. 0.10-0.25 on 3 of the 6 alternatives tested |
| **Speaker similarity** (Resemblyzer cosine similarity) | Cloning fidelity vs. the anchor clip | 0.94 same-language (EN), 0.85 cross-lingual (BM/ID) — quantifies the "accent bleed-through" the original proposal only flagged qualitatively |
| **VRAM / parameter count** | Light vs. heavy, measured not estimated | **2,384M params, 5.76 GB peak VRAM** — the heaviest model tested, confirming it runs at the edge of this hardware's 6GB budget |
| **Prosody delta** (F0/pacing with vs. without the control instruction) | Does emotion control actually do anything | Pitch variance nearly halves (39 vs. 68 Hz std), speaking rate drops ~20% — measurably works for tone/pacing; doesn't reliably increase pause frequency on the one sentence tested |

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

| Folder | Contents | Committed to git? |
|---|---|---|
| `src/` | VoxCPM2 generation scripts (baseline TTS + voice-cloning tests) | Yes |
| `src/comparison/` | One-off smoke-test scripts for each of the 9 alternative models | Yes |
| `src/eval/` | Objective evaluation: WER, speaker-similarity, VRAM/model-size profiling, prosody analysis | Yes |
| `results/audio/` | VoxCPM2 `.wav` test outputs (baseline, cloned-voice, and uncontrolled per language) | **Yes** — ~14 MB of audio, committed on purpose (see [postmortem](reports/poc_rerun_results.md#why-this-was-rebuilt-and-what-changed-to-prevent-it-happening-again)) |
| `results/comparison/` | `.wav` outputs and timing logs from the 9-model comparison | **Yes** — ~4 MB of audio, same reasoning |
| `results/eval/` | JSON output from the objective evaluation scripts | Yes (small, text-only) |
| `results/logs/` | Per-call generation timing logs for VoxCPM2 | Yes (small, text-only) |
| `reports/` | Written analysis — the model comparison, the original infra proposal, and the PoC rerun writeup | Yes |

Only `.venv/`, `.venvs/`, `.cache/`, and Python build artifacts are gitignored (see
[`.gitignore`](.gitignore)) — everything under `results/` is intentionally committed, not generated fresh
per clone. Total repo size is currently ~18-19 MB, almost entirely the `.wav` files above; not a problem
today, but if the audio corpus grows meaningfully beyond this scale, moving `results/audio/` and
`results/comparison/` to [Git LFS](https://git-lfs.com/) would be the next step rather than letting the
main history keep growing with binary blobs.

## Setup

Requires **Python 3.10–3.12**. GPU: NVIDIA, CUDA 12.0+ driver, 8GB+ VRAM recommended (this PoC was
run on a 6GB card — see the report for the tradeoffs that implies).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Why so many venvs

This repo uses **11 separate virtual environments**, all gitignored (`.venv/`, `.venvs/*/`) so none of
them are committed — only `requirements.txt` and `src/eval/requirements-eval.txt` are checked in.
They're split up because the 10 TTS models compared here (see [Why VoxCPM2](#why-voxcpm2)) come from
unrelated frameworks with genuinely incompatible pinned dependencies, not out of carelessness:

| Venv | Purpose |
|---|---|
| `.venv/` | Main project env — VoxCPM2 (`src/`), the model this repo builds on. `requirements.txt`. |
| `.venvs/eval/` | Objective evaluation (`src/eval/`) — Whisper, Resemblyzer, librosa. `src/eval/requirements-eval.txt`. |
| `.venvs/coqui/` | Coqui XTTSv2 (`src/comparison/run_coqui_xtts.py`) |
| `.venvs/styletts2/` | StyleTTS2 — needs `torch==2.5.1` pinned, older than every other venv here, because its checkpoint predates PyTorch 2.6's `weights_only=True` default and fails to load on newer torch |
| `.venvs/parler/` | Parler-TTS Mini |
| `.venvs/chattts/` | ChatTTS |
| `.venvs/f5tts/` | F5-TTS |
| `.venvs/piper/` | Piper |
| `.venvs/melo/` | MeloTTS install attempt (blocked — see comparison report) |
| `.venvs/fishspeech/` | Fish Speech install attempt (blocked — see comparison report) |

**Why not one shared venv:** each framework pins its own `torch`/`transformers`/`numpy` versions, and at
least one (StyleTTS2, pinned to `torch==2.5.1`) directly conflicts with the `torch==2.6.0+cu124` this
project's main `.venv` and most other comparison venvs use. Merging them would mean constantly
downgrading/upgrading torch between runs, or picking one version and having some models simply fail to
load — isolation is the actual fix here, not a workaround to clean up later. There's no single shared
`requirements.txt` for `src/comparison/`; each model's exact install steps and version pins are documented
per-model in [`reports/tts_models_comparison.md`](reports/tts_models_comparison.md#original-research-full-model-catalog).

To recreate any of them:

```bash
py -3.12 -m venv .venvs/<name>
.venvs\<name>\Scripts\activate      # Windows
# then follow that model's install notes in the comparison report
```

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

Objective evaluation (run from a dedicated `.venvs/eval/` — see the comparison report for setup):

```bash
python src/eval/run_wer.py                   # Whisper-based WER for every generated clip
python src/eval/run_speaker_similarity.py    # Resemblyzer cosine similarity for every cloned clip
python src/eval/run_prosody_analysis.py      # F0/pacing delta: control instruction vs. none
python src/eval/measure_vram_voxcpm2.py      # peak VRAM + param count (one script per model)
```

## Development

`src/text_data.py` is the single source of truth for every test sentence used across the
generation, comparison, and eval scripts — edit it there, not in the individual scripts.

Lint and unit tests are lightweight (no GPU or model downloads required) and run in CI on every
push/PR via [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

```bash
pip install -e ".[dev]"
ruff check .            # lint
ruff format --check .   # formatting
pytest                  # unit tests (tests/)
```

## Model attribution & licensing

Model weights are downloaded on first run (via Hugging Face) and are not redistributed in this repo.
Code licenses are the wrapper library's; model *weights* can carry a separate, stricter license —
check before any commercial use:

| Model | Wrapper library license | Notes |
|---|---|---|
| VoxCPM2 | Apache-2.0 | Weights and code both Apache-2.0 — the only fully permissive option here |
| Piper | GPL-3.0-or-later | Voice files (`rhasspy/piper-voices`) carry their own per-voice licenses |
| Coqui XTTSv2 | MPL-2.0 (`coqui-tts` library) | XTTSv2 model weights are under Coqui's non-commercial Public Model License (CPML), separate from the library's MPL-2.0 |
| StyleTTS2 | MIT | Community repackaging (`sidharthrajaram/StyleTTS2`), not the original authors' package |
| Parler-TTS Mini | Apache-2.0 | |
| ChatTTS | AGPLv3+ | Copyleft — network use may trigger source-disclosure obligations |
| F5-TTS | MIT (library) | Check the specific checkpoint's license before commercial use |

## References

### TTS models evaluated

- [VoxCPM2 (Hugging Face)](https://huggingface.co/openbmb/VoxCPM2) — the model selected for this project; Apache-2.0, emotion-controllable, native Malay/Indonesian/English support with zero-shot voice cloning
- [Piper](https://github.com/rhasspy/piper) — fast, lightweight VITS-based TTS with a large community voice catalog (`rhasspy/piper-voices`)
- [Coqui TTS / XTTSv2](https://github.com/idiap/coqui-ai-TTS) — multilingual (17 languages) autoregressive TTS with zero-shot voice cloning
- [StyleTTS 2](https://github.com/yl4579/StyleTTS2) — style-diffusion TTS with adversarial training
- [Parler-TTS](https://huggingface.co/parler-tts) — text-to-audio model conditioned on a natural-language voice description
- [ChatTTS](https://github.com/2noise/ChatTTS) — conversational TTS tuned for dialogue-style prosody
- [F5-TTS](https://github.com/SWivid/F5-TTS) — non-autoregressive flow-matching TTS
- [MeloTTS](https://github.com/myshell-ai/MeloTTS) — fast multilingual TTS (EN/ZH/JP/KR/FR/ES); not evaluated further, see [comparison report](reports/tts_models_comparison.md) for why
- [Fish Speech](https://github.com/fishaudio/fish-speech) — LLM-based TTS foundation model; not evaluated further, see comparison report
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — LLM + flow-matching TTS; not evaluated further, see comparison report

### Related tools and concepts

- [ElevenLabs](https://elevenlabs.io/) — commercial TTS API benchmarked against in the infrastructure proposal
- [torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html) — PyTorch's JIT graph compiler, evaluated in the PoC rerun
- [Triton](https://github.com/triton-lang/triton) / [triton-windows](https://github.com/woct0rdho/triton-windows) — the compiler backend `torch.compile` depends on for CUDA kernels; Windows builds are versioned separately from upstream Triton
- [FFmpeg](https://ffmpeg.org/) — required for voice-cloning reference-audio decoding

### Evaluation methods and metrics

- [Whisper (OpenAI)](https://github.com/openai/whisper) — used to transcribe generated audio back to text for Word Error Rate scoring
- [jiwer](https://github.com/jitsi/jiwer) — WER computation library
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) — speaker-embedding model used for voice-cloning similarity scoring
- [librosa](https://librosa.org/) — used for F0/pitch extraction and pause detection in the prosody-delta analysis
