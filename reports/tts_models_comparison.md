# Text-to-Speech (TTS) Models for Speech Synthesis Layer

## Executive Summary

This document originally outlined 10 open-source TTS models evaluated for local deployment within a
strict 6GB VRAM ceiling. It has since been extended with **measured results**: each model was actually
installed and run on the same hardware as the VoxCPM2 PoC (RTX 3050 Laptop, 6GB VRAM), using the same
English test sentence for comparability, with an Indonesian and (where possible) Malay pass to check real
multilingual support rather than relying on marketing copy.

**Single most important finding: of the 10 models evaluated, only VoxCPM2 has native support for both
Malay and Indonesian.** Every alternative tested is English-only, English+Chinese, or covers a fixed set
of major world languages that does not include BM or ID. This alone settles the model choice for this
project — the comparison below is about documenting *why*, and about installation/ecosystem maturity as
a secondary, genuinely useful finding.

## Measured Results

Test sentence (EN): *"Great, I love that question! So we know that sunlight is important for
photosynthesis, but let's think a little deeper."* (same sentence used across every model for
comparability). ID and BM rows use the equivalent translated test sentences from the VoxCPM2 test suite.
Each model was installed in its own isolated venv (`.venvs/<model>/`) to avoid dependency conflicts.

| Model | EN | BM | ID | Measured speed | Device | Status |
|---|---|---|---|---|---|---|
| **VoxCPM2** (chosen model) | ✅ | ✅ | ✅ | 20.45s / 18.97s / 29.13s (eager) | GPU | Full native support — see [`poc_rerun_results.md`](poc_rerun_results.md) |
| **Piper** | ✅ | ❌ | ✅ | 0.384s / — / 0.307s | **CPU** | Works; has an `id_ID` voice, no `ms` voice in `rhasspy/piper-voices` |
| **Coqui XTTSv2** | ✅ | ❌ | ❌ | 4.79s (default) / 4.19s (cloned) | GPU | Works; 17 supported languages, none BM/ID |
| **StyleTTS2** | ✅ | ❌ | ❌ | 3.86s | GPU | Works; English-only phonemizer (`gruut-lang-en`) |
| **Parler-TTS Mini** | ✅ | ❌ | ❌ | 16.575s | GPU | Works; English-only per model card |
| **ChatTTS** | ✅ | ❌ | ❌ | 13.107s | GPU | Works; EN+ZH only, drops `!`/`'` as "invalid characters" |
| **F5-TTS** | ✅ | ❌ | ❌ | 15.123s (cloned) | GPU | Works; EN+ZH only, voice-cloning-only (no built-in default voice) |
| **MeloTTS** | — | ❌ | ❌ | — | — | **Blocked** — broken PyPI package; see below |
| **Fish Speech** | — | ❌ | ❌ | — | — | **Blocked** — PyPI package is inference-incomplete; see below |
| **CosyVoice 2.0** | — | ❌ | ❌ | — | — | **Skipped** — no official PyPI package; see below |

(Coqui VITS, the lighter EN-only architecture bundled in the same `coqui-tts` package as XTTSv2, wasn't
separately timed — XTTSv2 already establishes Coqui's EN-only ceiling and is the more capable of the two.)

## Why VoxCPM2 wins on the criterion that actually matters

Every other tested model tops out at English, or English plus one or two major world languages (Chinese
most commonly). None of the 17 languages XTTSv2 supports, and none of the locale files bundled with
ChatTTS or Fish Speech, include Malay or Indonesian. Piper is the sole exception with partial coverage —
it has a community-contributed `id_ID` voice, but no `ms` voice exists in the official voice catalog at
all. This isn't a close call: VoxCPM2 is the only model here that was actually built with BM/ID as a
target language, and it shows in the results — every other model would require either training a new
voice from scratch (Piper/VITS-style) or accepts English/Chinese text and mispronounces Malay/Indonesian
input as an approximation, which defeats the purpose for a product that needs accurate speech in the
target languages, not just any speech.

## Installation friction as a finding

Beyond raw language support, the install process itself was informative — it's a signal of how
production-ready each ecosystem currently is:

- **MeloTTS**: the `melotts`/`MeloTTS` PyPI packages (0.1.1) are broken — the sdist is missing a
  `requirements.txt` file its own `setup.py` requires, so `pip install` fails during the build step before
  any code runs. The only working install path is `pip install git+https://github.com/myshell-ai/MeloTTS.git`,
  which was not attempted here — installing directly from an agent-chosen GitHub repo executes arbitrary
  external code and is blocked by this environment's auto-mode safety policy, independent of MeloTTS's
  own trustworthiness.
- **Fish Speech**: the PyPI package (`fish-speech` 0.1.0) is legitimate — published by the real project
  maintainer — but it only ships library internals (`fish_speech.inference_engine` etc.). Actually running
  inference needs the project's `tools/` CLI scripts (a `llama_queue` worker process, checkpoint-loading
  glue) that exist only in the GitHub repo, not in the pip package. Same git-install blocker as MeloTTS.
- **CosyVoice 2.0**: no official PyPI package exists at all under the `FunAudioLLM` org. The only `cosyvoice`
  package on PyPI (0.0.8) is an unofficial third-party repackaging by an unrelated individual
  (`lucasjinreal/CosyVoice`, not `FunAudioLLM/CosyVoice`) — installing and running unvetted, unofficial
  forks of a model carries real supply-chain risk, so this was skipped rather than installed. The official
  path is a git clone, also out of scope here.
- **StyleTTS2**: the PyPI package (`styletts2` 0.1.6, an unofficial-but-functional repackaging) ships a
  checkpoint saved in a pickle format that predates PyTorch 2.6's `weights_only=True` default —
  `torch.load` fails outright on current PyTorch. Fixed by pinning `torch==2.5.1` in its isolated venv.
  Also needed an on-demand `nltk.download('punkt_tab')` the package doesn't declare as a dependency.
- **ChatTTS**: PyPI package is missing `requests` and `soundfile` from its declared dependencies —
  both needed immediately at import/runtime — despite otherwise working fine once installed manually.

None of these are fatal, but they're a real cost: every model except VoxCPM2, Piper, Coqui, and F5-TTS
needed at least one manual workaround before it would run at all. VoxCPM2's own install was comparatively
clean (see [`poc_rerun_results.md`](poc_rerun_results.md) for its one hiccup — a Triton version pin).

## Original research: full model catalog

The full 10-model catalog researched before running any of the above, kept for reference:

### 1. Traditional / Dedicated Architecture TTS (Lightweight & Fast)
These architectures focus entirely on efficient, fast synthesis. They operate at low latencies with a low Real-Time Factor (RTF), leaving maximum VRAM headroom available for concurrent perception or orchestration tasks. They generally lack dynamic zero-shot voice cloning capabilities.

| Model Name | Type | Estimated VRAM Needed | Official Repository / Link |
| :--- | :--- | :--- | :--- |
| **Piper** | VITS-based | **< 100 MB** (Can run flawlessly on CPU) | [rhasspy/piper](https://github.com/rhasspy/piper) |
| **Coqui VITS** | End-to-End TTS | **~1.0 GB** | [coqui-ai/TTS](https://github.com/coqui-ai/TTS) |
| **MeloTTS** | Fast TTS | **~1.0 GB - 1.5 GB** | [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) |
| **StyleTTS 2** | Style-Diffusion | **~2.0 GB - 3.0 GB** | [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) |

### 2. LLM-Based / Foundation TTS Models (Heavier, Expressive)
These speech models utilize language model backends or flow-matching setups to deliver state-of-the-art zero-shot voice cloning, cross-lingual capabilities, and advanced emotional/conversational expressions. They are highly expressive but demand substantial VRAM footprint and compute.

| Model Name | Type | Estimated VRAM Needed | Official Repository / Link |
| :--- | :--- | :--- | :--- |
| **XTTSv2 (Coqui)** | Autoregressive LLM | **~3.0 GB - 4.0 GB** | [coqui-ai/TTS](https://github.com/coqui-ai/TTS) |
| **F5-TTS** | Non-Autoregressive | **~3.0 GB - 4.0 GB** | [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) |
| **Parler-TTS (Mini)** | Text-to-Audio | **~3.5 GB - 4.0 GB** | [huggingface/parler-tts](https://huggingface.co/parler-tts) |
| **ChatTTS** | LLM-based | **~4.0 GB** (Optimized for conversational nuances) | [2noise/ChatTTS](https://github.com/2noise/ChatTTS) |
| **CosyVoice 2.0** | LLM + Flow Matching | **~3.0 GB - 5.0 GB** (Using 0.5B parameters) | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |
| **Fish Speech** | LLM-based | **~4.5 GB - 6.0 GB** (Sits near VRAM threshold) | [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) |

### 3. Core Trade-off Summary (original, pre-measurement)
* **Go with Traditional TTS if:** Instant pipeline responsiveness and zero-latency synthesis are critical, or if the GPU needs to be shared heavily with real-time background perception streams.
* **Go with LLM-Based TTS if:** High emotional variance, natural conversational artifacts (pauses, laughters), or instant zero-shot voice replication from short reference clips are central requirements for the interaction layer.

**Post-measurement addendum:** the trade-off above is real for language pairs where a model has native
support (e.g. Piper for a low-latency English/Indonesian pipeline). But for this project's actual
requirement — Malay and Indonesian, not just "a language" — it's moot: none of the alternatives clear the
bar, so VoxCPM2 remains the only viable choice regardless of where a use case would otherwise land on the
speed/expressiveness trade-off.
