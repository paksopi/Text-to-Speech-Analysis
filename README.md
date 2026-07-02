# Text-to-Speech Analysis — Malay / Indonesian / English

## In-progress work (paused — resume later)

Goal: re-run the objective eval suite (WER, speaker similarity, VRAM, prosody) across *every* model,
including the 3 previously blocked ones, retried by actually installing from their official GitHub repos
(not just PyPI) — plus a few extra lightweight TTS models for a broader personal report, even ones that
don't meet this project's BM/ID requirement. License issues are fine to hit — document and move on
rather than stop.

**Status as of pause:**

- **Fish Speech** — re-investigated, and the earlier "PyPI package is inference-incomplete" finding is
  **wrong/outdated**: the pip package (`.venvs/fishspeech`) actually contains everything needed
  (`fish_speech.inference_engine.TTSInferenceEngine`, `launch_thread_safe_queue`, `load_decoder_model`) —
  only the `tools/` CLI wiring was missing, not the underlying library. The real, current blocker: the
  checkpoint (`fishaudio/openaudio-s1-mini` on Hugging Face) is **gated** — requires a logged-in HF
  account that has accepted the model's terms. License is the **Fish Audio Research License**
  (non-commercial/research use only; this project's evaluation use would qualify). **To resume:** provide
  an HF token from an account that has accepted the terms at
  huggingface.co/fishaudio/openaudio-s1-mini, then re-run the download + a minimal inference script
  wired the same way as the repo's `tools/run_webui.py`.
- **MeloTTS** — official git install (`pip install git+https://github.com/myshell-ai/MeloTTS.git`) was
  started in `.venvs/melo` and was mid-download (torch, transformers 4.27.4, tensorboard, etc. — pins an
  old `transformers`) when the session was paused; not yet complete, no install/license blocker hit yet.
  **To resume:** re-run that pip install to completion, then run a smoke-test EN generation.
- **CosyVoice 2.0** — cloned the official repo (`FunAudioLLM/CosyVoice`, Apache-2.0, incl. the
  `Matcha-TTS` submodule) into a scratch dir; created `.venvs/cosyvoice` (Python 3.12). Requirements pin
  `torch==2.3.1+cu121`; was mid-check on whether a cp312 wheel exists for that exact pin when paused —
  not yet resolved. No official PyPI package exists (unchanged from the original finding), so this still
  requires the git-clone path. **To resume:** finish the torch/cp312 compatibility check, `pip install -r
  requirements.txt` (with `--extra-index-url https://download.pytorch.org/whl/cu121`), and copy the
  cloned repo (or just its `cosyvoice/` + `third_party/Matcha-TTS/` dirs + `tools/`) somewhere durable,
  since it currently only exists in the session's scratch directory and will be lost.
- **New lightweight models identified (not yet installed):**
  - **MMS-TTS** (`facebook/mms-tts-zlm` for Malay, `facebook/mms-tts-ind` for Indonesian, via
    `transformers`' `VitsModel`) — small per-language VITS checkpoints, genuinely relevant since they
    actually claim Malay + Indonesian support, unlike everything except VoxCPM2/Piper so far. **License:
    CC-BY-NC 4.0** (non-commercial) — a license issue, but fine for this evaluation per the "just
    document it" instruction. `.venvs/mms` was created but dependency install (`torch`, `transformers`,
    `scipy`, `soundfile`) did not complete before pausing.
  - **Kokoro-82M** (`hexgrad/Kokoro-82M`) — very lightweight (82M params, ~350MB), Apache-2.0, no
    license issue. **Does not support Malay/Indonesian** (EN + a handful of others only) — doesn't meet
    this project's actual requirement, but worth running anyway per request for a broader personal
    report. `.venvs/kokoro` was created but nothing installed yet.
  - Considered and rejected: **Zonos-v0.1** (Zyphra) — no Malay/Indonesian support and not obviously
    lighter than models already covered, so lower priority than the two above.
- **Not started yet:** extending `src/eval/manifest.py` + the `measure_vram_*.py` scripts to cover
  whichever of the above end up runnable, re-running `run_wer.py` / `run_speaker_similarity.py` /
  `run_prosody_analysis.py`, and updating this README + `reports/tts_models_comparison.md` with the
  final results (or final blocker reasons) for all of the above.

**Note:** all installs above were paused mid-flight and the associated background processes were killed;
nothing above is left running. `.venvs/cosyvoice`, `.venvs/mms`, `.venvs/kokoro` exist but are mostly
empty (venv scaffold only, gitignored like all `.venvs/*`). The CosyVoice repo clone lives only in the
session's temp scratch directory, not in this repo — it'll need re-cloning next session.

[![CI](https://github.com/paksopi/Text-to-Speech-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/paksopi/Text-to-Speech-Analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

Comparative analysis and benchmarking of open-source text-to-speech (TTS) models for Malay (BM),
Indonesian (ID), and English (EN) speech synthesis with zero-shot voice cloning. 10 models were
researched; all 10 were actually installed and run (or documented as blocked) on the same hardware
(RTX 3050 Laptop, 6GB VRAM), using the same test sentences for comparability, and scored on four
objective metrics — WER, speaker similarity, VRAM/parameter count, and prosody delta — rather than
relying on marketing copy about language support.

Of the 10, **[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)** (Apache-2.0, open-weight) came out as
this project's selected model — see [Selected model](#selected-model-voxcpm2) below for why — and the
rest of this repo builds out its proof-of-concept in more depth: environment setup, model download,
EN/BM/ID generation tests, and zero-shot voice cloning.

- Full model comparison, measured results, and install-friction notes: [`reports/tts_models_comparison.md`](reports/tts_models_comparison.md)
- VoxCPM2 infrastructure/feasibility writeup — GPU sizing and cost vs. ElevenLabs: [`reports/VoxCPM2_PoC_Infrastructure_Proposal.md`](reports/VoxCPM2_PoC_Infrastructure_Proposal.md)
- VoxCPM2 PoC rerun results (after the original was lost and rebuilt): [`reports/poc_rerun_results.md`](reports/poc_rerun_results.md)

## Model Comparison

10 models were compared: VoxCPM2, Piper, Coqui XTTSv2, StyleTTS2, Parler-TTS Mini, ChatTTS, F5-TTS,
MeloTTS, Fish Speech, and CosyVoice 2.0. 3 of the 10 (MeloTTS, Fish Speech, CosyVoice) couldn't even be
installed cleanly — broken/incomplete PyPI packages, or no official package at all — which is itself part
of the finding, not a footnote; see the comparison report's [Installation Friction](reports/tts_models_comparison.md#installation-friction-as-a-finding)
section for what specifically broke on each.

| Model | EN | BM | ID | Measured speed | Mode | Device | Status |
|---|---|---|---|---|---|---|---|
| VoxCPM2 | ✅ | ✅ | ✅ | 20.45s / 18.97s / 29.13s | eager | GPU | Full native support |
| Piper | ✅ | ❌ | ✅ | 0.384s / — / 0.307s | default | CPU | Only other model with any ID support |
| Coqui XTTSv2 | ✅ | ❌ | ❌ | 4.79s / — / — | default | GPU | 17 languages, none BM/ID |
| StyleTTS2 | ✅ | ❌ | ❌ | 3.86s / — / — | default | GPU | English-only |
| Parler-TTS Mini | ✅ | ❌ | ❌ | 16.575s / — / — | default | GPU | English-only |
| ChatTTS | ✅ | ❌ | ❌ | 13.107s / — / — | default | GPU | EN+ZH only |
| F5-TTS | ✅ | ❌ | ❌ | 15.123s / — / — | default | GPU | EN+ZH only |
| MeloTTS | — | ❌ | ❌ | — | — | — | Blocked (broken PyPI package) |
| Fish Speech | — | ❌ | ❌ | — | — | — | Blocked (PyPI package inference-incomplete) |
| CosyVoice 2.0 | — | ❌ | ❌ | — | — | — | Skipped (no official PyPI package) |

**Only VoxCPM2 has native Malay support, and only VoxCPM2 + Piper support Indonesian** — every other
model tops out at English, or English+Chinese.

"Mode" = eager vs. `torch.compile`-compiled inference. VoxCPM2 is the only model here whose wrapper
exposes an explicit eager/compile toggle (`optimize=`); its number above is **eager**, since the PoC
rerun found `torch.compile` is 10-20x *slower* in this one-off-call test pattern — see
[VoxCPM2 Proof-of-Concept](#voxcpm2-proof-of-concept) below. "default" for the other 9 models means each
library's own out-of-the-box inference path was used as-is, with no `torch.compile` applied by this
project's scripts — but whether any of those libraries silently self-compile internally wasn't
independently verified, so those numbers should be read as "as installed," not confirmed-eager. **Known
limitation:** this eager/compiled distinction was only tracked for VoxCPM2; it's not recorded for the
other 9.

## Objective Evaluation

Four automated metrics back up the table above with numbers instead of vibes — full methodology and
results in the [comparison report's Objective Evaluation section](reports/tts_models_comparison.md#objective-evaluation).
**Preliminary, not statistically powered:** each number below comes from a single generation per
condition (n=1 per cell, one sentence for the prosody comparison) — useful for spotting large gaps, not
for fine-grained ranking between close scores; see the report's sample-size caveat for detail, and
[Limitations & Future Work](#limitations--future-work) below.

| Metric | What it measures | VoxCPM2 result |
|---|---|---|
| **WER** (Whisper transcription vs. requested text) | Does the model say the right words | **0.0000** across all 6 EN/BM/ID generations — vs. 0.10-0.25 on 3 of the 6 alternatives tested |
| **Speaker similarity** (Resemblyzer cosine similarity) | Cloning fidelity vs. the anchor clip | 0.94 same-language (EN), 0.85 cross-lingual (BM/ID) — quantifies the "accent bleed-through" the original proposal only flagged qualitatively |
| **VRAM / parameter count** | Light vs. heavy, measured not estimated | **2,384M params, 5.76 GB peak VRAM** — the heaviest model tested, confirming it runs at the edge of this hardware's 6GB budget |
| **Prosody delta** (F0/pacing with vs. without the control instruction) | Does emotion control actually do anything | Pitch variance nearly halves (39 vs. 68 Hz std), speaking rate drops ~20% — measurably works for tone/pacing; doesn't reliably increase pause frequency on the one sentence tested |

## Selected model: VoxCPM2

VoxCPM2 is the only one of the 10 compared models with native Malay support, and ties with Piper as the
only one with Indonesian support. That settles model choice for this project's actual language
requirement regardless of the speed/VRAM trade-offs above — full reasoning, architecture background, and
the full 10-model research catalog are in the [comparison report](reports/tts_models_comparison.md#why-voxcpm2-wins-on-the-criterion-that-actually-matters).
Its wrapper library and weights are both Apache-2.0 — see [Model attribution & licensing](#model-attribution--licensing)
for how that compares to the other 6 models that actually run.

## VoxCPM2 Proof-of-Concept

Rebuilt and reran on the same RTX 3050 Laptop (6GB VRAM) hardware as the original infrastructure
proposal, across two runs. All twelve test generations succeeded — EN/BM/ID baseline (default voice) and
EN/BM/ID cloned from a single anchor clip (the EN baseline output, reused as the zero-shot voice-cloning
reference across all three languages) — see the rerun report for the full test script/control-instruction
text.

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
compilation cost never amortized. This isn't a contradiction of the infrastructure proposal's RTX 4090
compile projections, which assume a production serving loop with many repeated, shape-stable calls; a
handful of one-off, varying-length test calls is close to the worst case for `torch.compile`. Practical
takeaway: keep `optimize=False` (eager mode) for exploratory testing like this, and only enable compile
once building a real serving loop. Full breakdown and open items in the rerun report.

## Infrastructure & Cost

The [infrastructure proposal](reports/VoxCPM2_PoC_Infrastructure_Proposal.md) sizes what self-hosting
VoxCPM2 actually costs beyond this dev laptop, benchmarked against ElevenLabs (the closest commercial
API alternative with comparable emotion/style control):

| GPU (VRAM) | Basis | Sec / call | \$ / call |
|---|---|---|---|
| RTX 3050 Laptop (6GB) — this repo's PoC | **Measured** | ~96.8 (avg of 3 langs) | n/a (owned hardware) |
| RTX 4090 (24GB), standard mode | Projected from official RTF 0.30 | ~6.7 | ~\$0.0006-0.0009 |
| RTX 4090 (24GB), Nano-vLLM accelerated | Projected from official RTF 0.13 | ~2.9 | ~\$0.0003-0.0004 |
| ElevenLabs API (Multilingual v2) | Commercial baseline | n/a | \$0.017-0.035 |

**Headline takeaway:** once running on adequately-sized hardware (8GB+ VRAM, ideally 24GB for comfortable
throughput), self-hosted VoxCPM2 is roughly 20-50x cheaper per call than the ElevenLabs API for
equivalent text length — before accounting for ElevenLabs' fixed monthly subscription floor. The
trade-off is operational, not financial: self-hosting means owning uptime, scaling, and serving
infrastructure instead of paying a vendor to manage it. Full GPU-tier breakdown, methodology, and the
"why the current laptop is unsuitable beyond PoC" reasoning are in the proposal doc.

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
| `reports/` | Written analysis — the model comparison, the infra proposal, and the PoC rerun writeup | Yes |

Only `.venv/`, `.venvs/`, `.cache/`, and Python build artifacts are gitignored (see
[`.gitignore`](.gitignore)) — everything under `results/` is intentionally committed, not generated fresh
per clone. Total repo size is currently ~18-19 MB, almost entirely the `.wav` files above; not a problem
today, but if the audio corpus grows meaningfully beyond this scale, moving `results/audio/` and
`results/comparison/` to [Git LFS](https://git-lfs.com/) would be the next step rather than letting the
main history keep growing with binary blobs.

## Setup

Requires **Python 3.10–3.12**. GPU: NVIDIA, CUDA 12.0+ driver, 8GB+ VRAM recommended (this PoC was
run on a 6GB card — see [Infrastructure & Cost](#infrastructure--cost) for the tradeoffs that implies).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Why so many venvs

This repo uses **11 separate virtual environments**, all gitignored (`.venv/`, `.venvs/*/`) so none of
them are committed — only `requirements.txt` and `src/eval/requirements-eval.txt` are checked in.
They're split up because the 10 TTS models compared here (see [Model Comparison](#model-comparison)) come
from unrelated frameworks with genuinely incompatible pinned dependencies, not out of carelessness:

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

## Limitations & Future Work

This analysis is honest about what it doesn't cover yet:

- **No human-rated naturalness score.** None of the four objective metrics (WER, speaker similarity,
  VRAM, prosody delta) capture naturalness or emotional appropriateness. A small MOS (Mean Opinion Score)
  pass — e.g. 5-10 listeners rating the 12 VoxCPM2 test clips on a 1-5 scale — would close this gap; see
  the comparison report's [Future Work](reports/tts_models_comparison.md#future-work-a-human-mos-pass)
  note.
- **Small sample sizes, no confidence intervals.** Every objective-evaluation number is from a single
  generation per condition (n=1 per cell; 1 sentence for the prosody comparison) — directional, not
  statistically powered. Expanding to 15-20 generations per language per model would require re-running
  inference across every model's isolated venv; flagged as follow-up, not attempted in this pass.
- **Eager-vs-compiled tracked for one model only.** `torch.compile` was found 10-20x slower than eager
  mode for VoxCPM2 in this test pattern, but that toggle isn't exposed the same way in the other 9
  libraries — their "default" speed numbers aren't confirmed eager or compiled.
- **Cross-lingual accent bleed-through is unassessed by native speakers.** The speaker-similarity metric
  quantifies that BM/ID clones score lower than the EN clone (0.85 vs. 0.94), but whether that gap is
  audibly an "accent" to a native Malay/Indonesian listener hasn't had human review.
- **Long-form stability is untested.** All generation tests here use short (~20-25 word) utterances;
  VoxCPM2's own documentation notes instability on longer text, which would need chunking and separate
  testing before assuming linear scaling.
- **No production serving layer.** Every result here is a manual, single-request generation call. A real
  serving layer (batching, concurrent requests, warm compiled-graph reuse) is unscoped — see the
  infrastructure proposal's "Serving architecture not yet scoped" item.

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
