# VoxCPM2 PoC Rerun — Results

Rebuild of the original PoC (see [`VoxCPM2_PoC_Infrastructure_Proposal.md`](VoxCPM2_PoC_Infrastructure_Proposal.md))
after the project folder was lost. Same machine as the original proposal (HP Victus 15, RTX 3050 Laptop
6GB VRAM), same target languages (EN / BM / ID), same `'calm, articulate, reflective'` control instruction.

## Environment actually installed

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch / torchaudio | 2.6.0+cu124 |
| voxcpm | 2.0.3 |
| triton-windows | 3.7.1.post27 |
| Model | `openbmb/VoxCPM2` (auto-downloaded from Hugging Face, ~1m27s over 9 files) |
| Denoiser | Disabled (`load_denoiser=False`), matching the original proposal's VRAM-saving mitigation |

**torch.compile / Triton:** installed and `optimize=True` was requested, but VoxCPM2 disabled it
internally at load time with: `torch.compile disabled - cannot import name 'AttrsDescriptor' from
'triton.compiler.compiler'` — a version mismatch between this `triton-windows` build and what VoxCPM2's
compiled kernels expect. Generation silently ran in eager mode. This is a different failure mode than
the original proposal anticipated (VRAM-driven OOM) — here it's a Triton/VoxCPM2 API compatibility gap,
not a memory ceiling. Worth revisiting with a pinned/older `triton-windows` version if the full
`torch.compile` speedup is needed.

## Baseline generation (default voice, no reference clip)

Test sentence: a one-sentence mentor-style line (~20-25 words per language), shorter than the original
proposal's 58-word script, so absolute times aren't directly comparable — but the relative EN/BM/ID
pattern held.

| Language | Time | Output |
|---|---|---|
| EN | 20.45s | `results/audio/en_baseline.wav` (8.16s audio) |
| BM | 18.97s | `results/audio/bm_baseline.wav` (7.36s audio) |
| ID | 29.13s | `results/audio/id_baseline.wav` (11.04s audio) |

Even in eager mode (no working `torch.compile`), these times are well under the original proposal's
92.9–102.7 sec/utterance — expected, since the test sentence here is roughly a third of the length.

## Voice-clone anchor test

Per-project decision: instead of recording a separate reference clip, `en_baseline.wav` from the run
above was reused as the **anchor** voice (`reference_wav_path`) and the same three sentences were
regenerated cloned from it — mirroring the original proposal's "one reference clip reused across all 3
languages" validation.

| Language | Time | Output |
|---|---|---|
| EN | 56.05s | `results/audio/en_cloned.wav` (8.16s audio) |
| BM | 45.59s | `results/audio/bm_cloned.wav` (8.96s audio) |
| ID | 49.49s | `results/audio/id_cloned.wav` (10.24s audio) |

Cloning roughly doubled-to-tripled generation time per call versus the uncloned baseline (reference
audio has to be encoded through the audio VAE each call), but all three cloned outputs are non-empty,
correctly durationed, and audibly carry the anchor voice's timbre across languages — replicating the
original proposal's cross-lingual timbre-transfer finding on a self-sourced anchor clip rather than a
separate recording.

## Open items carried over from the original proposal

- **Accent bleed-through** (EN-sourced anchor voiced in BM/ID) is unassessed here too — needs
  native-speaker review, same as the original proposal flagged.
- **Triton/torch.compile speedup is still unvalidated** on this hardware — the compatibility issue above
  means the production-speed path (RTF ~0.13–0.30 on better GPUs) hasn't actually been exercised locally.
- Long-form stability, language quality parity, and serving-layer scoping are unchanged from the original
  proposal's §6 — see that document for full detail.
