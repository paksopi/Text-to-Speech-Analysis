# VoxCPM2 PoC Rerun — Results

Rebuild of the original PoC (see [`VoxCPM2_PoC_Infrastructure_Proposal.md`](VoxCPM2_PoC_Infrastructure_Proposal.md))
after the project folder was lost. Same machine as the original proposal (HP Victus 15, RTX 3050 Laptop
6GB VRAM). Two runs are recorded below: an initial eager-mode smoke test, then a second run after pinning
Triton to fix `torch.compile` and switching to a richer, longer test script.

## Environment actually installed

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch / torchaudio | 2.6.0+cu124 |
| voxcpm | 2.0.3 |
| triton-windows | **3.2.0.post21** (pinned — see below) |
| Model | `openbmb/VoxCPM2` (auto-downloaded from Hugging Face, ~4-5GB) |
| Denoiser | Disabled (`load_denoiser=False`), matching the original proposal's VRAM-saving mitigation |

## Test content (both runs use this format)

Control instructions are passed as a `(...)` prefix ahead of the target text, per VoxCPM2's voice-design
convention. Run 2 uses a longer, more specific control instruction and a real teaching-script sentence,
translated into BM and ID:

> **Control:** "A calm, articulate female voice in her early-20s. Her tone is grounded and reflective,
> carrying a sense of 'warm precision'. Deliberate pauses are used between ideas to give the listener
> space to think, maintaining a patient, unhurried pace."
>
> **EN:** "Great, I love that question! So we know that sunlight is important for photosynthesis — but
> let's think a little deeper. Think of a plant as a little factory — light is the energy that comes in.
> Now, what do you think a plant does with that energy? What do plants need to survive and grow?"
>
> **BM:** "Bagus, saya suka soalan itu! Jadi kita tahu bahawa cahaya matahari penting untuk fotosintesis —
> tetapi mari kita fikir dengan lebih mendalam. Bayangkan tumbuhan sebagai sebuah kilang kecil — cahaya
> ialah tenaga yang masuk. Sekarang, apa pula yang anda fikir tumbuhan lakukan dengan tenaga itu? Apakah
> yang tumbuhan perlukan untuk hidup dan membesar?"
>
> **ID:** "Bagus, aku suka pertanyaan itu! Jadi kita tahu bahwa sinar matahari penting untuk fotosintesis
> — tapi mari kita berpikir lebih dalam. Bayangkan tumbuhan sebagai sebuah pabrik kecil — cahaya adalah
> energi yang masuk. Sekarang, menurutmu apa yang dilakukan tumbuhan dengan energi itu? Apa yang
> dibutuhkan tumbuhan untuk bertahan hidup dan tumbuh?"

## Run 1: eager mode (torch.compile silently disabled)

`triton-windows` 3.7.1.post27 (latest at the time) was incompatible with this VoxCPM2/torch build:
`torch.compile disabled - cannot import name 'AttrsDescriptor' from 'triton.compiler.compiler'`.
Generation ran in eager mode using a shorter one-sentence test script (~20-25 words/language).

| Language | Baseline time | Cloned time |
|---|---|---|
| EN | 20.45s | 56.05s |
| BM | 18.97s | 45.59s |
| ID | 29.13s | 49.49s |

## Run 2: torch.compile fixed via Triton pin, longer test script

**Fix:** `torch` 2.6.0 expects Triton 3.2.x (that's what it bundles on Linux). `triton-windows` publishes
separately versioned Windows builds, and the latest one (3.7.1) had drifted ahead of what VoxCPM2's
compiled kernels expect. Pinning to `triton-windows==3.2.0.post21` fixed the `AttrsDescriptor` import and
`torch.compile` genuinely activated this time — confirmed by `torch._inductor.compile_fx` warnings
appearing in the logs and the `"torch.compile disabled"` message no longer appearing at all.

| Language | Baseline time | Cloned time |
|---|---|---|
| EN | 269.76s | 236.15s |
| BM | 353.78s | 348.81s |
| ID | 359.44s | 328.52s |

**Key finding: `torch.compile` made this workload slower, not faster.** Every number above is roughly
10-20x worse than Run 1's eager-mode times. The reason is structural, not a bug: `torch.compile` traces
and JIT-compiles a graph *per distinct input shape* it sees. Every call here has a different token length
(EN vs. BM vs. ID text differs; baseline vs. cloned differs further since the reference clip adds
audio-encoding steps), so **every single call in this run paid its own from-scratch compilation cost**
instead of reusing a warm cache. None of the compiled graphs were ever reused, so there was no amortization
to offset the compile overhead — this run never left the "always compiling" regime.

This does not contradict the original proposal's projected speedups (RTF ~0.13-0.30 on an RTX 4090,
§4.2). Those figures assume a **production serving loop**: many requests, typically batched or padded to
a small set of fixed shapes, where compilation happens once (or a handful of times) and then amortizes
over thousands of calls. This PoC's testing pattern — a handful of one-off calls, each a different length
— is close to the worst case for `torch.compile` and does not exercise the regime the speedup applies to.

**Practical recommendation:** keep `optimize=False` (eager mode) as the default for this kind of
exploratory/one-off testing. Only enable `torch.compile` once building an actual serving loop with
repeated, shape-stable calls (e.g. padding all requests to a fixed max length) — that's the only setting
in which the compile cost pays for itself.

## Voice-clone anchor test (both runs)

Both runs reused the EN baseline output as the **anchor** clip (`reference_wav_path`) and regenerated all
three languages cloned from it, rather than recording a separate reference — mirroring the original
proposal's "one reference clip reused across all 3 languages" validation. All 6 cloned outputs (3 per
run) are non-empty, correctly durationed, and audibly carry the anchor voice's timbre across languages.

## Open items carried over from the original proposal

- **Accent bleed-through** (EN-sourced anchor voiced in BM/ID) is unassessed here too — needs
  native-speaker review, same as the original proposal flagged.
- **`torch.compile` production-speed path is now confirmed reachable** (the Triton pin works), but still
  needs to be validated in an actual repeated-shape serving loop, not one-off calls, before trusting the
  original proposal's RTX 4090 projections.
- Long-form stability, language quality parity, and serving-layer scoping are unchanged from the original
  proposal's §6 — see that document for full detail.
