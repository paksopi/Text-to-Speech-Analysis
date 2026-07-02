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

Beyond language support, four objective metrics were run against the generated audio (see [Objective
Evaluation](#objective-evaluation) below): **Word Error Rate** (0.0% for VoxCPM2 across all six EN/BM/ID
generations — every word transcribed correctly, vs. 10-25% WER on three of the six alternatives tested),
**speaker-similarity** for voice cloning (0.85-0.94 cosine similarity, with cross-lingual clones scoring
measurably lower than same-language ones — a real number behind the "accent bleed-through" concern the
original proposal only flagged qualitatively), **VRAM/parameter-count profiling** (VoxCPM2 is honestly
the heaviest model tested, 2.4B params / 5.76GB peak VRAM — the real cost of its BM/ID support), and a
**prosody-delta analysis** of the emotion control instruction (it measurably flattens pitch variance and
slows speaking rate as intended, though doesn't reliably increase pause frequency on the one sentence
tested).

## Measured Results

Test sentence (EN): *"Great, I love that question! So we know that sunlight is important for
photosynthesis, but let's think a little deeper."* (same sentence used across every model for
comparability). ID and BM rows use the equivalent translated test sentences from the VoxCPM2 test suite.
Each model was installed in its own isolated venv (`.venvs/<model>/`) to avoid dependency conflicts.

| Model | EN | BM | ID | Measured speed | Mode | Device | Status |
|---|---|---|---|---|---|---|---|
| **VoxCPM2** (chosen model) | ✅ | ✅ | ✅ | 20.45s / 18.97s / 29.13s | **eager** | GPU | Full native support — see [`poc_rerun_results.md`](poc_rerun_results.md) |
| **Piper** | ✅ | ❌ | ✅ | 0.384s / — / 0.307s | default | **CPU** | Works; has an `id_ID` voice, no `ms` voice in `rhasspy/piper-voices` |
| **Coqui XTTSv2** | ✅ | ❌ | ❌ | 4.79s (default) / 4.19s (cloned) | default | GPU | Works; 17 supported languages, none BM/ID |
| **StyleTTS2** | ✅ | ❌ | ❌ | 3.86s | default | GPU | Works; English-only phonemizer (`gruut-lang-en`) |
| **Parler-TTS Mini** | ✅ | ❌ | ❌ | 16.575s | default | GPU | Works; English-only per model card |
| **ChatTTS** | ✅ | ❌ | ❌ | 13.107s | default | GPU | Works; EN+ZH only, drops `!`/`'` as "invalid characters" |
| **F5-TTS** | ✅ | ❌ | ❌ | 15.123s (cloned) | default | GPU | Works; EN+ZH only, voice-cloning-only (no built-in default voice) |
| **MeloTTS** | — | ❌ | ❌ | — | — | — | **Not attempted** — git-based install blocked by this environment's install policy; see below |
| **Fish Speech** | — | ❌ | ❌ | — | — | — | **Blocked** — library installs and runs, but the checkpoint is gated on Hugging Face; see below |
| **CosyVoice 2.0** | — | ❌ | ❌ | — | — | — | **Not attempted** — git-based install blocked by this environment's install policy; see below |
| MMS-TTS *(extra, not part of the 10-model catalog)* | — | ✅ | ✅ | 0.961s / 0.274s | default | CPU | Works; Malay + Indonesian, CC-BY-NC 4.0 (non-commercial) |
| Kokoro-82M *(extra, not part of the 10-model catalog)* | ✅ | ❌ | ❌ | 1.342s | default | GPU | Works; English-only, no BM/ID support |

(Coqui VITS, the lighter EN-only architecture bundled in the same `coqui-tts` package as XTTSv2, wasn't
separately timed — XTTSv2 already establishes Coqui's EN-only ceiling and is the more capable of the two.)

**Eager vs. compiled, and why it matters here:** the PoC rerun (see [`poc_rerun_results.md`](poc_rerun_results.md))
found `torch.compile` is **10-20x slower** than eager mode for VoxCPM2 in this test pattern — a swing large
enough to change which model looks fastest, so every "Measured speed" number above is labeled by mode.
VoxCPM2's `optimize=` toggle (in `src/common.py`) is the only explicit eager/compile switch this project's
own code exposes; its number is confirmed **eager**. The other 9 models were run through each library's
own out-of-the-box call path with no `torch.compile` wrapping applied by this project — labeled
**default** rather than "eager" because it wasn't independently verified whether any of those libraries
compile internally by default. **Known limitation:** the eager-vs-compiled distinction that mattered so
much for VoxCPM2 was not tracked per-model for the other 9; their "default" numbers should be read as
"as installed, not benchmarked under both modes," not as a confirmed apples-to-apples eager comparison.

## Objective Evaluation

Beyond "does it support the language," four automated metrics were run against the generated audio to
put numbers behind quality claims instead of relying on listening tests. Scripts live in `src/eval/`,
raw output in `results/eval/`. True MOS (human-rated naturalness) isn't automatable and isn't claimed
here — these four are the objective proxies that are.

**Sample-size caveat — read before citing these numbers elsewhere:** every metric below is computed on a
single generation per condition (6 VoxCPM2 clips total: EN/BM/ID × baseline/cloned; 1-5 clips for the
alternative models; 1 sentence pair for the prosody-delta comparison). These are **directional findings,
not statistically powered results** — there is no repeat-generation variance data, so no confidence
interval can honestly be reported (an interval computed from n=1 per cell would be meaningless, not
just imprecise). A single bad/good transcription or a single unlucky Resemblyzer embedding shifts a
reported number entirely, with no way to tell signal from generation-to-generation noise. Treat every
WER/similarity/prosody number here as "this is what happened on the one run we generated," useful for
spotting large, obvious gaps (e.g. VoxCPM2's 0.0 WER vs. F5-TTS's 0.25), not for precise ranking between
models that land close together. Expanding to 15-20 generations per language per model — enough to
report a rough confidence interval — would require re-running inference across every model's isolated
venv (see [Why so many venvs](../README.md#why-so-many-venvs)), which wasn't done as part of this pass;
flagged here as **preliminary, unpowered results** and as follow-up work rather than silently presented
as settled.

### 1. Word Error Rate (WER) — does the model actually say the right words

Each generated clip was transcribed back with Whisper (`medium`, language pinned per clip) and diffed
against the text that was actually requested (lowercased, punctuation-stripped). This is the closest
analogue to the language-detection repo's accuracy metric: 0.0 = every word transcribed matches exactly.

| Model | Lang | File | WER |
|---|---|---|---|
| VoxCPM2 | en/ms/id | baseline ×3 | **0.0000** / **0.0000** / **0.0000** |
| VoxCPM2 | en/ms/id | cloned ×3 | **0.0000** / **0.0000** / **0.0000** |
| Piper | en/id | — | **0.0000** / **0.0000** |
| Coqui XTTSv2 | en | default / cloned | **0.0000** / **0.0000** |
| Parler-TTS Mini | en | — | **0.0000** |
| ChatTTS | en | — | 0.1000 |
| StyleTTS2 | en | — | 0.1500 |
| F5-TTS | en (cloned) | — | 0.2500 |
| MMS-TTS | ms/id | — | 0.0784 / **0.0000** |
| Kokoro-82M | en | — | **0.0000** |

VoxCPM2 has zero transcription errors across all six EN/BM/ID generations, baseline and cloned alike —
this is a real correctness signal, not just "it sounds fine": Whisper heard exactly the requested words
in every one of VoxCPM2's outputs, in three languages. F5-TTS's 0.25 WER is the model's own known
instability on short reference-cloned generations (it substituted "Lovely question" for "Great, I love
that question" — a plausible-sounding but wrong transcription, not a minor slip). Full transcriptions in
[`results/eval/wer_results.json`](../results/eval/wer_results.json).

### 2. Speaker similarity — how good is the voice cloning, numerically

Cosine similarity between a Resemblyzer speaker-embedding of the anchor clip and of each cloned output.
1.0 = identical embedding, ~0.0 = unrelated voices; real same-speaker recordings under different
conditions typically land around 0.75–0.95.

| Model | Clone | vs. anchor | Similarity |
|---|---|---|---|
| VoxCPM2 | en_cloned (same language as anchor) | en_baseline | **0.9361** |
| VoxCPM2 | bm_cloned (cross-lingual) | en_baseline | 0.8532 |
| VoxCPM2 | id_cloned (cross-lingual) | en_baseline | 0.8496 |
| Coqui XTTSv2 | en_cloned | en_baseline | 0.9448 |
| F5-TTS | en_cloned | en_baseline | 0.9496 |

MMS-TTS and Kokoro-82M have no rows here — neither was run in voice-cloning mode (MMS-TTS is a
fixed-checkpoint VITS model with no cloning support at all; Kokoro was only smoke-tested with its default
voice), so there's no cloned-vs-anchor comparison to score.

This quantifies something the original proposal only flagged qualitatively (§6, "accent bleed-through...
not yet formally assessed"): VoxCPM2's cross-lingual clones (BM/ID from an English anchor) score
measurably lower on speaker similarity than its same-language EN clone (0.85 vs. 0.94) — real evidence,
not just a suspicion, that cloning across languages costs some voice fidelity. It's still a strong score
(comparable to XTTSv2's and F5-TTS's same-language numbers), just not as tight as same-language cloning.

### 3. VRAM footprint & model size — light vs. heavy, measured

Peak `torch.cuda.max_memory_allocated()` during one generation call, parameter counts where measurable,
and on-disk checkpoint size (first-run download size). Piper is CPU-only (no VRAM used at all).

| Model | Params (measured) | Peak VRAM (measured) | Checkpoint size (disk) |
|---|---|---|---|
| VoxCPM2 | **2,384M** | **5.76 GB** | 4.7 GB |
| Parler-TTS Mini | 878M | 4.29 GB | 3.3 GB |
| Coqui XTTSv2 | 467M | 2.01 GB | 1.8 GB |
| StyleTTS2 | 70M+ (partial — decoder+predictor only) | 1.59 GB | 0.87 GB |
| ChatTTS | not exposed by the library | 1.24 GB | 1.2 GB |
| F5-TTS | not measured — see note | not measured — see note | 1.3 GB |
| Piper | N/A (ONNX, CPU) | **0 GB (CPU-only)** | ~61 MB per voice |
| Kokoro-82M | 81.8M | 0.73 GB | ~350 MB |
| MMS-TTS | 36.3M | 0.47 GB | ~145 MB per language |

VoxCPM2 is, honestly, the heaviest model tested — 2.4B parameters and 5.76 GB peak VRAM, which matches
the original proposal's measured "~5.8/6.0GB" almost exactly and confirms it's genuinely running at the
edge of this hardware's 6GB budget, not comfortably inside it. This is the real cost behind VoxCPM2's
BM/ID support and zero WER: it's a substantially larger model than any of the EN-only alternatives (5-30x
more parameters than StyleTTS2 or Piper). F5-TTS's VRAM probe segfaulted intermittently on this hardware
after a long session of loading/unloading many models back-to-back — not reproduced as a clean number,
noted here rather than guessed at.

### 4. Prosody delta — does the control instruction actually do anything

VoxCPM2's emotion/style control works by prefixing the text with a natural-language instruction (e.g.
*"A calm, articulate female voice... Deliberate pauses are used between ideas..."*). To check whether
this measurably changes the output rather than being cosmetic, the same EN sentence was generated twice
from the same model — once with that control instruction, once with none — and compared on pitch (F0)
and pacing via `librosa`.

| Metric | With control instruction | No control instruction |
|---|---|---|
| F0 (pitch) mean | 190.25 Hz | 202.85 Hz |
| **F0 (pitch) std dev** | **39.29 Hz** | **67.76 Hz** |
| Speaking rate | 3.64 words/sec | 4.56 words/sec |
| Pause ratio | 0.157 | 0.229 |
| Pause count | 34 | 39 |

Two of the four metrics clearly support the instruction's intent: pitch variance nearly halves (39.29 vs.
67.76 Hz std) — a flatter, calmer pitch contour, consistent with "calm" and "grounded" — and speaking
rate drops about 20% (3.64 vs. 4.56 words/sec), consistent with "patient, unhurried pace." The pause
ratio result is a genuine miss, though: despite the instruction explicitly asking for "deliberate pauses,"
the *controlled* version actually paused *less* (15.7% vs. 22.9% of total duration) than the uncontrolled
one. Reported as-is rather than only citing the two metrics that confirm the story — the control
instruction measurably works for pitch and pacing, but doesn't reliably produce more/longer pauses on
this one test sentence.

### Future work: a human MOS pass

None of the four objective metrics above — WER, speaker similarity, VRAM/params, prosody delta — capture
**naturalness or emotional appropriateness**, which matters directly for a tutoring-context voice (the
project's actual use case). A clip can transcribe perfectly (0.0 WER) and still sound robotic, or clone a
voice with high cosine similarity and still sound flat or emotionally mismatched to "patient, encouraging
tutor." Closing this gap needs a small human-rated Mean Opinion Score (MOS) pass — e.g. 5-10 listeners
rating the 12 existing VoxCPM2 test clips (EN/BM/ID × baseline/cloned) on a 1-5 naturalness/appropriateness
scale — which is out of scope for this pass but flagged here as the clear next step before treating the
objective metrics above as a complete quality picture.

## Model background — what each architecture actually is

Not measured, but useful context for *why* the numbers above look the way they do:

| Model | Architecture | Training language coverage |
|---|---|---|
| **VoxCPM2** | Tokenizer-free, context-aware TTS — an LLM-style backbone paired with a diffusion-based audio decoder, no discrete phoneme/token intermediate step | Explicitly multilingual, including Malay and Indonesian as first-class targets — this is *why* it's the only model here with real BM/ID support, not an accident of scale |
| **Piper** | VITS (single-stage, GAN-based, non-autoregressive) — small, fast, one model per voice | Per-voice training data; community-contributed voices exist for ~30+ languages including Indonesian, but not Malay |
| **Coqui XTTSv2** | GPT-2-style autoregressive text-to-codes model + HiFi-GAN vocoder | 17-language multilingual training set (major European + East Asian languages), no Southeast Asian languages beyond none |
| **StyleTTS2** | Two-stage: acoustic text-to-style encoder + style-diffusion sampler + adversarial (GAN) decoder training | Trained on LJSpeech/LibriTTS — English only |
| **Parler-TTS Mini** | Encoder-decoder transformer (MusicGen-derived architecture), conditioned on a natural-language *description* of the voice rather than a reference clip | Trained on English-only description/audio pairs |
| **ChatTTS** | GPT-style autoregressive LLM backbone tuned for conversational/dialogue prosody | Bilingual EN/ZH training data |
| **F5-TTS** | Diffusion Transformer (DiT), non-autoregressive flow-matching — generates the full mel-spectrogram in one denoising pass conditioned on a reference clip | English + Mandarin training data |

The pattern across every non-VoxCPM2 model is the same: architecture choice (autoregressive vs.
diffusion vs. flow-matching) affects speed and expressiveness, but *training-data language coverage* is
what actually gates BM/ID support, and none of these were trained with Malay or Indonesian in scope.
VoxCPM2's advantage isn't a smarter architecture — several of these alternatives are architecturally
comparable or newer — it's that BM/ID were an explicit training target from the start.

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
- **Fish Speech**: the PyPI package (`fish-speech` 0.1.0), installed in `.venvs/fishspeech`, is legitimate
  and — contrary to an earlier finding in this project — is inference-complete: it exposes
  `fish_speech.inference_engine.TTSInferenceEngine`, `launch_thread_safe_queue`, and `load_decoder_model`,
  everything the official `tools/run_webui.py` wires together; only the `tools/` CLI convenience scripts
  are missing from the pip package, not the underlying library. The real blocker is the checkpoint itself:
  `fishaudio/openaudio-s1-mini` on Hugging Face is **gated**, requiring a logged-in account that has
  accepted the model's terms (Fish Audio Research License, non-commercial/research use). Resuming this
  needs a human to provide an HF token from an account that has accepted those terms.
- **CosyVoice 2.0**: no official PyPI package exists at all under the `FunAudioLLM` org. The only `cosyvoice`
  package on PyPI (0.0.8) is an unofficial third-party repackaging by an unrelated individual
  (`lucasjinreal/CosyVoice`, not `FunAudioLLM/CosyVoice`) — installing and running unvetted, unofficial
  forks of a model carries real supply-chain risk, so this was skipped rather than installed. The official
  path is a git clone, which is likewise blocked by this environment's auto-mode safety policy against
  installing/building from an agent-chosen external repo.
- **MMS-TTS** and **Kokoro-82M** (not part of the original 10-model catalog; added afterward for a
  broader personal report — see [Original research: full model catalog](#original-research-full-model-catalog)):
  both installed cleanly from official PyPI/Hugging Face packages (`transformers`' `VitsModel` for MMS-TTS,
  the `kokoro` package for Kokoro) with no workarounds needed. MMS-TTS's weights carry a CC-BY-NC 4.0
  (non-commercial) license; Kokoro-82M is Apache-2.0.
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
