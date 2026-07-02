# Text-to-Speech (TTS) Models for Speech Synthesis Layer

## Executive Summary
This document outlines 10 open-source Text-to-Speech (TTS) models evaluated for local deployment within a strict 6GB VRAM hardware ceiling. Building on the baseline established during testing of VoxCPM2, these alternatives range from ultra-fast traditional architectures to expressive LLM-based speech foundation models.

## 1. Traditional / Dedicated Architecture TTS (Lightweight & Fast)
These architectures focus entirely on efficient, fast synthesis. They operate at low latencies with a low Real-Time Factor (RTF), leaving maximum VRAM headroom available for concurrent perception or orchestration tasks. They generally lack dynamic zero-shot voice cloning capabilities.

| Model Name | Type | Estimated VRAM Needed | Official Repository / Link |
| :--- | :--- | :--- | :--- |
| **Piper** | VITS-based | **< 100 MB** (Can run flawlessly on CPU) | [rhasspy/piper](https://github.com/rhasspy/piper) |
| **Coqui VITS** | End-to-End TTS | **~1.0 GB** | [coqui-ai/TTS](https://github.com/coqui-ai/TTS) |
| **MeloTTS** | Fast TTS | **~1.0 GB - 1.5 GB** | [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) |
| **StyleTTS 2** | Style-Diffusion | **~2.0 GB - 3.0 GB** | [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) |

## 2. LLM-Based / Foundation TTS Models (Heavier, Expressive)
These speech models utilize language model backends or flow-matching setups to deliver state-of-the-art zero-shot voice cloning, cross-lingual capabilities, and advanced emotional/conversational expressions. They are highly expressive but demand substantial VRAM footprint and compute.

| Model Name | Type | Estimated VRAM Needed | Official Repository / Link |
| :--- | :--- | :--- | :--- |
| **XTTSv2 (Coqui)** | Autoregressive LLM | **~3.0 GB - 4.0 GB** | [coqui-ai/TTS](https://github.com/coqui-ai/TTS) |
| **F5-TTS** | Non-Autoregressive | **~3.0 GB - 4.0 GB** | [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) |
| **Parler-TTS (Mini)** | Text-to-Audio | **~3.5 GB - 4.0 GB** | [huggingface/parler-tts](https://huggingface.co/parler-tts) |
| **ChatTTS** | LLM-based | **~4.0 GB** (Optimized for conversational nuances) | [2noise/ChatTTS](https://github.com/2noise/ChatTTS) |
| **CosyVoice 2.0** | LLM + Flow Matching | **~3.0 GB - 5.0 GB** (Using 0.5B parameters) | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |
| **Fish Speech** | LLM-based | **~4.5 GB - 6.0 GB** (Sits near VRAM threshold) | [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) |

## 3. Core Trade-off Summary
* **Go with Traditional TTS if:** Instant pipeline responsiveness and zero-latency synthesis are critical, or if the GPU needs to be shared heavily with real-time background perception streams.
* **Go with LLM-Based TTS if:** High emotional variance, natural conversational artifacts (pauses, laughters), or instant zero-shot voice replication from short reference clips are central requirements for the interaction layer.
