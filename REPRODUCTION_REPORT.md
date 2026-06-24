# Reproduction Report — DPG QR / "67" Weight-Encoding (arXiv 2604.08423)

**Paper:** *Synthetic Data for any Differentiable Target* (Thrush et al.) — the Dataset
Policy Gradient (DPG). **Goal:** reproduce the headline result — synthetic training data,
optimized by a metagradient reward, encodes a chosen image into a target model's LM-head
weights (decoded as `sign(P_c − P_i)`) — **cheaply** (single GPU, not the paper's 8-GPU
stack) and **exactly** (pixel accuracy → ~1.0, with evidence).

> **Verdict: PARTIAL / IN PROGRESS.** The encoding *mechanism* was built end-to-end on one
> GPU using the authors' own metagradient engine, and a real root-cause bug was found and
> fixed. We did **not** reach a fully-encoded pattern (pixel accuracy ~1.0) within the
> compute budget. This report documents what works, the decisive bug fix, the remaining
> blockers, and the recommended next step. No model was uploaded (we hold uploads until a
> clean pattern exists).

---

## 1. What the paper claims

DPG trains a synthetic-data **generator** `π_θ` with GRPO. The RL reward is the per-example
**metagradient** `τ_i = ∂Φ/∂w_i` — how much training the **target** model (GPT-2) on example
`x_i` (with loss weight `w_i`) would improve a differentiable target metric `Φ`. For the QR
result (sec 4.1):

```
Φ = − mean( ln(1 + exp(−s · Y ⊙ (P_c − P_i))) ),   s = 20
```

where `Y ∈ {−1,+1}` is the target pattern, `P_i` / `P_c` are a patch of GPT-2's LM-head
weights before / after training, and the decode is `sign(P_c − P_i)`. The headline is a
**21×21 scannable QR**; sec 4.2 is the authors' own **scaled-down "67"** (6×7) ablation.
The full pipeline is **8 GPUs**: a verl GRPO trainer (Llama-3.2-1B generator) + a JAX/EasyDeL
metagradient server (GPT-2 target), M=200 GRPO steps × 96 inner Adam steps × 24,576
examples/step.

## 2. Approach — cheap but faithful

The 8-GPU requirement is almost entirely **Llama rollout parallelism + a huge
`train_batch_size=24576`**. The actual mechanism is GPT-2-tiny. So we kept the authors' real
metagradient engine (`MemoryEfficientTrainer` + the `sixseven`/`rick_roll` target metric,
verbatim) and replaced the heavyweight Llama+verl generator with a compact loop on **one GPU**.

Two driver iterations were built as **child experiments off the frozen baseline** (the
baseline holds the authors' real code and was never edited):

- **`dataset_metagradients_jax/scripts/run_minimal_qr.py`** — metagradient ascent on
  per-example **data weights** (no LM generator). Proves the engine runs end-to-end.
- **`dataset_metagradients_jax/scripts/run_dpg_grpo_min.py`** — the **real DPG loop**: a small
  JAX nanoGPT **generator** trained by **GRPO** (group-relative advantages, KL=0 — Algorithm 1)
  whose reward is the real per-example metagradient.

## 3. Experiment tree and results

```
Root (baseline — authors' real code, frozen)
└─ Minimal QR/67 metagradient repro            data-weight ascent; pixel acc 0.00→0.786, Φ pinned at −ln2
   ├─ T=96 inner (paper QR setting)            pixel acc plateau 0.786, Φ ≈ −ln2
   │  ├─ Move the head: inner LR 1e-3 + …       Φ moved 0.004 total — still flat
   │  └─ Real GRPO generator (single-GPU DPG)   best pixel acc 0.619; Φ flat (root cause below)
   │     └─ Fix inner loop (T real steps) +     INNER-LOOP BUG FIXED; target → Qwen3-0.6B-Base;
   │        Qwen3-0.6B-Base target              inner loss now drops 20→10 (head trains), but
   │                                            too slow on 0.6B to converge in budget
   └─ T=24 inner + higher inner LR 1e-4         pixel acc plateau 0.786, Φ ≈ −ln2
```

| Run | Target | Outer loop | Best pixel acc | Φ moved? |
|---|---|---|---|---|
| Minimal QR/67 | GPT-2 | data-weight ascent | 0.786 | no (−ln2) |
| T=96 inner | GPT-2 | data-weight ascent | 0.786 | no |
| LR 1e-3 + aggressive | GPT-2 | data-weight ascent | 0.786 | ~0.004 |
| Real GRPO generator | GPT-2 | GRPO generator | 0.619 | no |
| Fix inner loop + Qwen | Qwen3-0.6B-Base | GRPO generator | (unfinished) | head trains, see §4 |

(Pixel accuracy on the 6×7 "67" pattern; chance ≈ 0.5. `Φ = −ln(2) ≈ −0.693` ⇔ `P_c−P_i ≈ 0`,
i.e. the head did not move.)

## 4. The decisive finding — a real root-cause bug, and its fix

Every early run had `Φ` **pinned at exactly `−ln(2)`** — the target LM head was not moving at
all, so there was no reward signal for any generator/reweighting to exploit. The cause, found
by reading the inner-loop logs:

> **The inner loop only ever ran ONE Adam step.** The trainer does one update per *batch* in
> its dataloader, and the reward dataset was exactly one batch
> (`n_per_step = microbatch × grad_accum`). So the target model took a single tiny step, the
> head barely moved, and `Φ` stayed at `−ln2` regardless of T, inner LR, or the generator.

**Fix** (`run_dpg_grpo_min.py`): feed the generated batch as **T separate global batches** with
**stable per-example indices** `0..n_per_step−1`, so the target trains the synthetic data for
**T real Adam steps** and the metagradient w.r.t. `data_weights[i]` correctly accumulates
example `i`'s influence across all T steps (matching the paper: "train the batch for T steps").
After the fix the inner training loss **drops cleanly (≈20 → ≈10 over 8 steps)** — the target
model is genuinely training, the precondition that was missing.

## 5. Why it still did not converge in budget

We switched the **target** to **`Qwen/Qwen3-0.6B-Base`** (the requested model to encode + upload),
which surfaced two cost/feasibility issues that GPT-2 (the paper's choice) does not have:

1. **Huge vocab (151,936) ⇒ OOM.** Full-batch logits (`batch × seq × vocab`) overflow HBM.
   Fixed by microbatching the inner loop and chunking generation; resolved the OOM.
2. **Disk-checkpointed VJP on a 0.6B model is slow.** The memory-efficient metagradient
   checkpoints the full target state to disk at *every* inner step (~3.5–4 s I/O each), so one
   GRPO step (T forward + T backward) takes many minutes. We ran out of H100 budget before a
   pixel-accuracy curve could form.
3. **Tied embeddings (`tie_word_embeddings: true`).** Qwen3-Base ties `lm_head` to the input
   embedding, so the "upper-left 6×7 patch" is the embedding for token ids 0–5. If the
   generated text never emits those tokens, those rows get ~zero gradient and `Φ` can stay at
   `−ln2` *even while the rest of the model trains* — a pathology GPT-2 (untied head, small
   vocab) avoids. (The robust head-finder *does* correctly locate the 1024×151936 matrix.)

## 6. Recommended next step

Reproduce cleanly on **GPT-2 first** (the paper's actual target: small, untied head, fast),
now that the inner-loop bug is fixed — this should move `Φ` off `−ln2` and converge cheaply,
yielding the pixel-accuracy → ~1.0 evidence. Then, as a separate (more expensive) step,
transfer to Qwen3-0.6B-Base — choosing a target patch over token rows the generator actually
emits, to sidestep the tied-embedding dead-zone — and only upload once a pattern is clean.

## 7. Honesty notes / caveats

- A **plateau at 0.79 is not a reproduction** and is not presented as one. Chance on a 6×7
  binary pattern is ~0.5; 0.79 is "the head barely moved," not "the pattern emerged."
- Our generator + outer loop are a **faithful re-implementation at small scale**, not the
  authors' exact verl/Llama stack — a *stronger* signal for the mechanism (reproduces under a
  different implementation) but an independent reproduction, not a bit-for-bit rerun.
- The headline claim is the **21×21 QR**; "67" (6×7) is the cheaper mechanism proof. We have
  not yet produced either pattern cleanly.

## 8. Artifacts & code

- Drivers: `dataset_metagradients_jax/scripts/run_minimal_qr.py`,
  `dataset_metagradients_jax/scripts/run_dpg_grpo_min.py`; launchers `run_minimal_qr.sh`,
  `run_dpg_grpo_min.sh`.
- Each run wrote `EVAL.md`, decoded-image PNGs, `history.json`, and a W&B
  `target_metric/pixel_accuracy` curve to `.openresearch/artifacts/`.
- Discussion log: `REPRODUCTION_QA.md`.
