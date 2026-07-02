**Local Open-Weight TTS PoC: VoxCPM2**

_Infrastructure & Feasibility Brief - Malay / Indonesian / English, Emotion-Controllable TTS_

Prepared for project proposal review | PoC status: validated locally on consumer hardware

# 1\. Executive Summary

We evaluated VoxCPM2, an Apache-2.0 licensed open-weight text-to-speech model, as a candidate for a product requiring Malay (BM), Indonesian (ID), and English (EN) speech synthesis with emotion/style control. A working proof-of-concept was successfully run on a consumer laptop GPU (RTX 3050 Laptop, 6GB VRAM), confirming the model loads and generates correct, emotion-steerable speech in all three target languages, including zero-shot voice cloning from a short reference clip.

The PoC also surfaced a hard constraint worth flagging early: generation speed on this hardware is far below real-time (roughly 90-105 seconds to synthesize a ~20-second utterance). This is a hardware/configuration limitation, not a quality or correctness issue - the same model on a single mid-range cloud GPU is expected to run 15-25x faster. Section 4 lays out the upgrade path and cost.

# 2\. What Was Validated in the PoC

| **Capability**                                  | **Result**                                                        |
| ----------------------------------------------- | ----------------------------------------------------------------- |
| Model loads on consumer GPU (6GB VRAM)          | Confirmed - runs at ~5.8/6.0GB VRAM utilization                   |
| English generation with emotion control         | Confirmed - Voice Design control instructions applied correctly   |
| Malay (BM) generation with emotion control      | Confirmed                                                         |
| Indonesian (ID) generation with emotion control | Confirmed                                                         |
| Zero-shot voice cloning (single reference clip) | Confirmed - one reference clip reused across all 3 languages      |
| Cross-lingual timbre transfer                   | Confirmed working; accent bleed-through not yet formally assessed |
| Commercial licensing                            | Apache 2.0 - free for commercial use, no vendor contact required  |

Sample test: a ~58-word, four-sentence mentor-style teaching script, translated and voiced in all three languages using an identical 'calm, articulate, reflective' control instruction, to validate that voice character holds consistently across languages.

# 3\. Performance Results - Current Dev Hardware

## 3.1 Test machine

| **Component**                     | **Spec**                                                         |
| --------------------------------- | ---------------------------------------------------------------- |
| Laptop                            | HP Victus 15                                                     |
| CPU                               | AMD Ryzen 5 8645HS                                               |
| GPU                               | NVIDIA RTX 3050 Laptop, 6GB dedicated VRAM                       |
| iGPU                              | AMD Radeon 760M (not used - no CUDA/ROCm path for this workload) |
| VRAM utilization during inference | 5.8 / 6.0 GB (97%, GPU-bound)                                    |
| GPU utilization during inference  | 100%                                                             |

## 3.2 Generation time (per utterance, ~20s of output audio)

| **Language**    | **Wall-clock time** |
| --------------- | ------------------- |
| English (EN)    | 94.7 sec            |
| Malay (BM)      | 102.7 sec           |
| Indonesian (ID) | 92.9 sec            |

This works out to a Real-Time Factor (RTF) of roughly 5-6x slower than the audio's own duration. For comparison, the model's official benchmark reports an RTF of ~0.30 on an RTX 4090 (i.e., generation finishes in under a third of the audio's playback time). The gap is explained by three compounding factors specific to this dev setup, not by the model itself:

- **torch.compile disabled:** GPU acceleration (torch.compile) is disabled in this PoC to stay within the 6GB VRAM budget - re-enabling it requires more headroom than this card has.
- **Raw compute ceiling:** the RTX 3050 has roughly 5x fewer CUDA cores and roughly a third of the memory bandwidth of an RTX 4090.
- **Memory pressure:** running at 97% VRAM utilization leaves no room for the optimizations (CUDA graphs, larger batch/cache buffers) that unlock faster inference paths.

# 4\. Infrastructure Options for Production / Pilot

Output quality is not expected to change meaningfully with better hardware - a GPU upgrade buys speed and stability, not better-sounding audio. This section lays out the GPU options available, their expected speed and cost, and a direct cost comparison against ElevenLabs, the closest commercial API alternative, so the trade-off can be evaluated like-for-like.

## 4.1 Minimum viable spec (matches official model requirements)

| **Requirement** | **Value**                                           |
| --------------- | --------------------------------------------------- |
| VRAM            | 8 GB minimum (model's stated requirement)           |
| GPU             | NVIDIA, CUDA 12.0+ driver                           |
| Python          | 3.10 - 3.12                                         |
| PyTorch         | 2.5+ (CUDA-enabled build matching the GPU's driver) |
| Disk            | ~10-15 GB for model weights + dependencies          |
| RAM (system)    | 16 GB recommended                                   |

Note: our PoC ran on 6GB by disabling torch.compile acceleration and the optional denoiser - this brought the footprint under 8GB, but at the direct cost of the speed numbers below. See Section 3 for the mechanism.

## 4.2 GPU speed & cost comparison - apples-to-apples

_The table below distinguishes MEASURED results (our own test) from ESTIMATED results (projected from the model's own published benchmark figures). Estimates are flagged clearly and should be treated as planning guidance, not guaranteed performance - actual throughput should be re-verified on rented hardware before final budget sign-off._

**Reference test sentence: 56 words / 289 characters (~22.4 sec of output audio at a natural speaking pace).**

| **GPU (VRAM)**                         | **Basis**                        | **Sec / call**         | **Calls / hr** | **Cloud \$/hr** | **\$ / call**    |
| -------------------------------------- | -------------------------------- | ---------------------- | -------------- | --------------- | ---------------- |
| RTX 3050 Laptop (6GB) - our PoC        | MEASURED                         | ~96.8 (avg of 3 langs) | ~37            | owned hardware  | n/a (local)      |
| RTX 4060 (8GB)                         | ESTIMATED                        | ~60-75                 | ~50-60         | \$0.07-0.17     | ~\$0.001-0.003   |
| RTX 4070 (12GB)                        | ESTIMATED                        | ~35-45                 | ~80-100        | \$0.10-0.25     | ~\$0.001-0.003   |
| A10 (24GB)                             | ESTIMATED                        | ~30-40                 | ~90-120        | \$0.09-0.30     | ~\$0.001-0.003   |
| L4 (24GB)                              | ESTIMATED                        | ~20-30                 | ~120-180       | \$0.39-1.30     | ~\$0.002-0.006   |
| RTX 4090 (24GB), standard mode         | PROJECTED from official RTF 0.30 | ~6.7                   | ~536           | \$0.34-0.50     | ~\$0.0006-0.0009 |
| RTX 4090 (24GB), Nano-vLLM accelerated | PROJECTED from official RTF 0.13 | ~2.9                   | ~1,236         | \$0.34-0.50     | ~\$0.0003-0.0004 |
| L40S (48GB), production serving        | ESTIMATED, concurrent requests   | <3 (batched)           | 1,500+         | \$0.50-1.77     | ~\$0.0003-0.001  |

_RTX 4090 rows are the most reliable estimates in this table because they are derived directly from VoxCPM2's own published benchmark RTF (Real-Time Factor), not from a generic hardware scaling assumption. The RTX 4060/4070/A10/L4 rows are rougher order-of-magnitude estimates based on relative compute and should be confirmed with an actual rented session before relying on them for capacity planning._

## 4.3 Cost comparison vs. ElevenLabs API

ElevenLabs is the most relevant commercial benchmark since it is the market-leading hosted TTS API with comparable emotion/style control. As of mid-2026, ElevenLabs API pricing on the Multilingual v2 model runs approximately \$0.06-\$0.12 per 1,000 characters, with Flash/Turbo models at roughly half that rate but reduced quality and control depth.

|                                           | **ElevenLabs API (Multilingual v2)**                                        | **Self-hosted VoxCPM2 (RTX 4090)**                     |
| ----------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------ |
| Cost basis                                | \$0.06 - \$0.12 per 1,000 characters                                        | Cloud GPU \$/hr divided by calls/hr                    |
| Cost for our 289-char test sentence       | \$0.017 - \$0.035 per call                                                  | \$0.0003 - \$0.0009 per call                           |
| Cost for 10,000 calls/month (this length) | \$173 - \$347 / month                                                       | \$3 - \$9 / month (GPU compute only)                   |
| Fixed monthly cost                        | Subscription tier required (\$22-\$330+/mo for commercial + cloning rights) | \$0 (pay only for compute time actually used)          |
| Voice cloning                             | Included on Creator tier (\$22/mo) and above                                | Included - Controllable & Hi-Fi cloning, no extra tier |
| Licensing for commercial use              | Requires paid tier (\$5/mo minimum)                                         | Apache 2.0 - free, no tier or attribution required     |

**Headline takeaway: once running on adequately-sized hardware (8GB+ VRAM, ideally 24GB for comfortable throughput), self-hosted VoxCPM2 is roughly 20-50x cheaper per call than the ElevenLabs API for equivalent text length, before accounting for ElevenLabs' fixed monthly subscription floor. The trade-off is operational: we own the uptime, scaling, and serving infrastructure instead of paying a vendor to manage it.**

_This comparison excludes engineering time to build and maintain the serving layer (see Section 6, 'Serving architecture not yet scoped'), egress/storage costs, and ElevenLabs' broader feature set (90+ language dubbing, larger pre-built voice library, managed infrastructure, SLA-backed uptime) which has its own value depending on the product's needs._

## 4.4 Why the current laptop is unsuitable beyond PoC

- 6GB VRAM is below the model's stated 8GB requirement; the PoC only fits by disabling acceleration features.
- Triton/torch.compile (the acceleration layer behind every speed figure in 4.2) needs extra VRAM headroom this card does not have - it cannot simply be 'turned on' on this hardware. Speed is gated by VRAM, not just enabled by a setting.
- No headroom for concurrent requests - this setup can serve one user at a time, sequentially.
- Any other GPU-using application (browser hardware acceleration, screen sharing, etc.) risks an out-of-memory crash mid-generation.
- Recommendation: this hardware is sufficient to prove the concept (done) but not to pilot with real users, and not sufficient to validate real-world speed at all.

# 5\. Software Environment Checklist

Required for any machine (dev or cloud) running this PoC:

| **Component**                    | **Version / Notes**                                                               |
| -------------------------------- | --------------------------------------------------------------------------------- |
| Python                           | 3.10 - 3.12 (3.13+ and 3.14 are NOT supported by current dependencies)            |
| Virtual environment              | venv (isolates project dependencies from system Python)                           |
| PyTorch + torchaudio             | 2.5+, CUDA build matching the target GPU's driver                                 |
| voxcpm (pip package)             | Installs the model wrapper, CLI, and Python API                                   |
| FFmpeg                           | Required for any voice-cloning workflow (reference audio decoding via torchcodec) |
| Triton (Windows: triton-windows) | REQUIRED for production-speed inference (see note below) - not merely optional    |
| Model weights                    | openbmb/VoxCPM2 - auto-downloaded from Hugging Face on first run (~4-5GB)         |

_Important: Triton is the dependency that enables torch.compile, which is the ONLY path to the production-speed figures (RTF ~0.13-0.30) cited in Section 4. The current dev-laptop PoC runs WITHOUT Triton/torch.compile, specifically because the 6GB VRAM budget cannot absorb its extra compilation/warm-up memory overhead on top of the base model. This is why the PoC's measured times (94.7-102.7 sec/utterance) are 300-700x slower than the model's official benchmark, not a discrepancy in the model itself. On 8GB+ VRAM hardware, Triton should be installed and torch.compile enabled as a default, not an optional extra - Section 4's cost/speed table assumes this is in place._

# 6\. Open Risks & Follow-Up Items

- **Voice cloning accent quality:** Confirmed working, but cross-lingual accent bleed (e.g. an English-recorded reference voice sounding accented when cloned into Malay/Indonesian output) has not yet been formally assessed with native-speaker review.
- **Language quality parity:** Malay and Indonesian are lower-resource languages in the model's training mix than English or Chinese; quality should be validated against native-speaker feedback before committing to a launch language list.
- **Long-form stability:** Current testing used short (~20s) utterances. The model's own documentation notes instability on long-form text; production use cases involving longer scripts should be tested with chunking before assuming linear scaling.
- **Licensing:** Apache 2.0 is permissive and commercial-use-ready; no action needed, but worth a one-line confirmation from legal/compliance for the record.
- **Serving architecture not yet scoped:** This PoC tested manual single-request generation. Production deployment requires building a serving layer (e.g. via Nano-vLLM-VoxCPM for concurrent request handling) which has not yet been scoped.

# 7\. Recommendation

The PoC de-risks the core technical question: VoxCPM2 can produce emotion-controllable, natural-sounding speech in all three required languages, under a commercially-friendly license, and it runs - even if slowly - on minimal consumer hardware. The path to a usable pilot is a hardware change, not a model or architecture change, which is a low-risk, well-understood next step (provision one cloud GPU instance in the 8-24GB tier).

Suggested next step: provision a single RTX 4090-class cloud instance for one week, re-run this same test suite with torch.compile enabled, and gather native-speaker quality feedback on the Malay and Indonesian output before scaling further.