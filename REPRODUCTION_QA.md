# Reproduction Discussion — Q&A Log

This document records the working discussion around reproducing arXiv **2604.08423**
(*Synthetic Data for any Differentiable Target*, Thrush et al.) — the Dataset Policy
Gradient (DPG) QR-code / "67" weight-encoding result. It is a verbatim Q&A log of the
questions asked during the effort and the answers given, preserved as a record of the
reasoning, the decisions, and the implications.

---

## Q1. Can you get a minimal reproduction of arXiv 2604.08423 running end to end?

(Work log — the trajectory taken in response.)

- **Oriented on the project via `orx`:** one root/baseline node, no prior runs, repo
  `69mannying/dataset-policy-gradients-56b37a2b`. The baseline already holds the **authors'
  real implementation** (a JAX/EasyDeL metagradient server + a verl GRPO trainer), designed
  for an **8-GPU node**.
- **Read the paper.** The headline mechanism (DPG): a synthetic-data **generator** (`π_θ`,
  Llama-3.2-Instruct) is trained with GRPO; the RL reward is the per-example **metagradient**
  `τ_i = ∂Φ/∂w_i` — how much training the target model on example `x_i` would improve a
  differentiable target metric `Φ`. The QR result (sec 4.1) encodes a **21×21** scannable QR
  into GPT-2's LM-head weights via `Φ = -mean(ln(1 + e^{-s·Y⊙(P_c−P_i)}))`, s=20, decoded as
  `sign(P_c − P_i)`. Sec 4.2 is the authors' own **scaled-down "67"** (6×7) ablation.
- **Assessed the minimal path.** The full QR run is M=200 GRPO steps × 96 inner Adam steps ×
  24,576 examples/step on 8 GPUs — not minimal. The *core mechanism* (metagradient engine +
  `sixseven`/`rick_roll` target) has a standalone test harness needing no verl/RPC.
- **Built a minimal single-GPU driver** (`run_minimal_qr.py`) reusing the authors'
  `MemoryEfficientTrainer` + the `sixseven` target metric on GPT-2, with metagradient ascent
  on per-example **data weights** as a stand-in outer loop, created as a **child** of the
  (frozen) baseline. It ran **end to end**: env built, GPT-2 loaded, metagradient computed,
  decoded image emitted, `EVAL.md` + artifacts written.
- **Result:** pixel accuracy moved 0.000 → **0.786** but plateaued, with `Φ` pinned at
  `-ln(2)` — i.e. the LM-head patch barely moved. Diagnosis: **re-weighting a fixed pool of
  generic examples cannot move GPT-2's LM head far enough to flip signs** — it lacks the
  degree of freedom the paper's *generator* has (synthesizing new, pattern-aligned text).

---

## Q2. Make sure we reproduce the QR code part — WANDB_API_KEY is set right?

Confirmed: `orx env` shows **`WANDB_API_KEY (org)`** is set at the org level, so W&B logging
works and the `target_metric/pixel_accuracy` curve is available as evidence. Re-read the QR /
"67" methodology and hyperparameters from the paper to build the QR part faithfully (s=20,
GPT-2 target, Adam inner optimizer, 6×7 patch for "67" / 21×21 for QR, decode = sign(P_c−P_i)).

---

## Q3. Can we upload the resulting model to HF?

Two issues were flagged:

1. **The run was still in flight** — no final model yet from the round-2 run.
2. **More fundamentally: this minimal repro never produces a model worth uploading.** The
   driver resets the target GPT-2 to its pretrained init at the *start of every outer step*
   (because `A` is treated as a function — the paper's design). The metagradient `τ` is
   computed and returned as the **reward**, but the driver only updates the per-example data
   weights with it — it never persists a trained target model or the optimized generator. It
   writes `EVAL.md` + decoded-image PNGs, not a checkpoint. So there is no model artifact.

Options offered: (1) save & upload the final target GPT-2 (needs a save step + `HF_TOKEN` +
repo id; pattern only ~0.79, not clean); (2) the trained **generator**, which needs the
fuller verl GRPO pipeline; (3) finish the in-flight run first, then decide. Recommendation:
(3) then (1).

---

## Q4. (HF account `https://huggingface.co/reneeice`, token pasted) — upload to this.

**Security note given:** a HuggingFace access token was pasted in plaintext and should be
treated as compromised — **revoke/rotate it** at https://huggingface.co/settings/tokens. A
token must never be baked into a file (it would be committed to git and synced — a leak); it
belongs in the run env as a secret.

**Blocker surfaced:** uploading from inside a run needs `HF_TOKEN` in the run environment.
`orx env` only *lists* names; it cannot set values. Secrets are added in the OpenResearch
**web dashboard** (project/org → environment variables) — not via the CLI.

**And the result caveat reiterated:** the model this minimal repro produces is a GPT-2 whose
LM-head "67" pattern only reached **~0.79 pixel accuracy** — *partially* encoded, not a clean
QR/"67". A faithful, fully-encoded model needs the generator loop that was skipped.

---

## Q5. Wait — we need exact reproduction, not partial. Get the best possible outcome and good evidence before we upload.

Agreed: **exact reproduction before any upload.** The minimal data-reweighting loop plateaus
at 0.79 precisely because it lacks the generator. Re-read the authors' real pipeline
(`run_dpg_local.sh` + the qr/67 configs): it is genuinely an **8-GPU, ~200-GRPO-step** job —
Llama-3.2-1B generator (vLLM rollouts + FSDP) + the JAX GPT-2 metagrad server co-located on
one 8-GPU node, each step computing metagradients through 96 GPT-2 Adam steps over 24,576
synthetic examples. Multi-hour, plus building the verl venv from prebuilt CUDA wheels and
pulling LFS prompt data. Plan reset around getting the pattern to *actually emerge*
(pixel acc → ~1.0) with solid evidence.

---

## Q6. That's too expensive — optimize on it.

Key realization: **exact reproduction does not need 8 GPUs or the full verl stack.** The
8-GPU requirement is just how the authors *parallelized the generator*; the underlying claim
is reproducible far cheaper.

- **Target model = GPT-2 (124M).** Tiny. The metagradient inner loop (96 Adam steps on a 6×7
  LM-head patch) is the real "science" and it's cheap.
- **The expensive part is the generator** = Llama-3.2-1B doing vLLM rollouts + FSDP training
  of 24,576 examples/step × 200 steps — ~98% of the cost, most of it the huge
  `train_batch_size=24576`.
- The single-GPU loop plateaued at 0.79 **not** because of compute, but because
  **reweighting a fixed generic pool can't move the head** — it lacks the generator's degree
  of freedom (synthesizing new aligned tokens).
- The paper's own "67" ablation (sec 4.2) runs with **B=256/1024** and explicitly demonstrates
  the **small-batch regime** reaching ~perfect pixel accuracy.

**Optimized plan:** keep the real metagradient mechanism, replace the heavyweight Llama+verl
generator with the minimal faithful thing that supplies the missing degree of freedom — a
**small JAX GRPO generator** (nanoGPT) sampling token sequences, scored by the **real**
`MemoryEfficientTrainer` + `sixseven` metagradient reward, updated with GRPO (group-relative
advantages, KL=0 — Algorithm 1). Implemented as `run_dpg_grpo_min.py` and launched on a
**single GPU**.

---

## Q7. Let's remind ourselves — what's the goal here, and what about the QR?

**The goal.** Reproduce the headline result of 2604.08423: the Dataset Policy Gradient can
optimize a synthetic-data generator so that training GPT-2 on its data **encodes a chosen
image into the LM-head weights**, decoded as `sign(P_c − P_i)`. Two pattern results:

- **QR code (sec 4.1)** — the headline figure: a **21×21** scannable QR. Expensive
  (M=200 × 96 inner × 24,576 examples, 8 GPUs).
- **"67" (sec 4.2)** — the authors' own **scaled-down** version of the *same* mechanism (6×7,
  smaller M/B), introduced explicitly as the cheaper setting "to run more experiments," shown
  reaching ~perfect pixel accuracy.

Constraints set: **exact reproduction, not partial** (pixel acc → ~1.0, real evidence) and
**keep it cheap** (no 8-GPU full run).

**How those reconcile — and where the QR fits.** QR and "67" are the *same algorithm* on a
different target pattern `Y`. Strategy:

1. **Nail the mechanism cheaply on "67" first** — get `sign(P_c − P_i)` to converge to the 6×7
   "67" at ~1.0 pixel accuracy on one GPU. This *is* a faithful, exact reproduction of sec 4.2.
2. **Then scale the same converged loop to the 21×21 QR** — only the target array `Y` and the
   patch size change (21×21 vs 6×7), plus more GRPO steps. If "67" converges, QR is the same
   code with a bigger pattern and more compute; decide the extra cost then.

So **the QR is not abandoned** — "67" is the cheap proof the encoding mechanism works, and QR
is the scale-up of that exact loop. The driver keeps the pattern as a swappable target
(`TARGET_67` now; a 21×21 `Y` later).

---

## Q8. What are the implications of this if proved with evidence?

### What the proof actually establishes

The narrow technical claim sounds small — "you can make a QR code appear in GPT-2's weights."
But the *mechanism* that makes it possible is the real result, and it generalizes:

1. **Training data can be optimized to install *arbitrary differentiable properties* into a
   model — precisely, and via ordinary SFT.** The QR/"67" is a stress test chosen because it's
   unambiguous: a pattern in the weights either decodes correctly or it doesn't. If a generator
   can steer 42 specific weights to specified signs *through nothing but the data the model
   trains on*, the same metagradient machinery can target any differentiable metric — a
   benchmark loss, a behavior, an internal representation. The image is a thermometer; the
   disease is "data → arbitrary model internals, controllably."

2. **It makes the metagradient → policy-gradient approximation empirically real, not just a
   theorem.** The paper's Theorem 3.1 says `τ_i = ∂Φ/∂w_i` approximates the true (intractable)
   dataset-level policy gradient. A clean QR is the strongest empirical confirmation: an
   end-to-end behavior that *only* succeeds if that approximation is accurate enough to
   optimize through 96 steps of Adam. An independent reproduction (different code path) hardens
   the claim.

3. **It turns "data attribution" from analysis into control.** Influence functions / attribution
   ask "which data caused this behavior?" This flips it: "what data would *cause* a chosen
   behavior?" — and answers it constructively. A qualitative shift from diagnosing models to
   programming them through their training set.

### Implications, by who cares

- **Security / data poisoning (the sharp edge):** a constructive recipe for *targeted, stealthy*
  poisoning. The generated text still looks like normal Wikipedia rephrases (coherence is
  preserved), yet it deterministically installs a chosen weight pattern — a backdoor, a bias, a
  hidden trigger — that a casual eval wouldn't surface. "Looks benign, trains malicious" *with
  precision* is a meaningfully scarier threat model than random poisoning. Independent
  reproduction means it's not an artifact of one lab's setup.

- **Provenance / watermarking / steganography:** you can deliberately *write* a recoverable
  signature into a model's weights through its training data — a watermark or ownership stamp
  that survives standard fine-tuning, or covert channels in weight space. The QR being literally
  *scannable* is the point: the encoded payload is exact and readable.

- **Alignment / capability steering (the constructive side):** the same tool can inject
  knowledge, enforce a style/register, or push internal representations toward a target —
  systematically, by curating data rather than hand-designing losses. A knob for "shape the
  model via what it reads."

- **Data attribution theory:** the paper's other finding — **Adam, not SGD, in the inner loop is
  essential** — says optimizer dynamics matter for attribution far more than the standard
  (SGD/influence-function) theory assumes. If our reproduction also shows the inner-LR / Adam
  dependence, that's independent evidence that attribution methods ignoring optimizer state are
  mis-specified.

### Honest caveats on *our* evidence specifically

- **"67" at ~1.0** proves the *encoding mechanism* (sec 4.2). It does **not** by itself prove the
  headline **21×21 QR** — a harder target (441 constrained weights vs 42) and the figure people
  remember. To claim "we reproduced the QR result," we need the QR target, not just "67."
- Our generator and outer loop are a **faithful re-implementation at small scale**, not the
  authors' exact verl/Llama stack. That's a *stronger* result for the *mechanism* (reproduces
  under a different implementation → not setup-specific), but it's an independent reproduction,
  not a bit-for-bit rerun. Worth stating plainly.
- A clean curve from random-init pixel accuracy monotonically to ~1.0, plus the decoded image
  matching `Y`, plus the Adam-dependence, is genuinely convincing evidence. **A plateau at 0.8
  is not a reproduction and should not be dressed up as one.**

**Takeaway if proven with evidence:** *synthetic training data is a precise, general, and
stealthy control surface for model internals — confirmed by independent reproduction.* That is
why the paper frames it as much a safety/poisoning warning as a capability, and why insisting on
the full QR (not a partial pattern) is the right call: the dramatic, defensible version of the
claim *is* the scannable QR.

---

## Status at time of writing

- Minimal data-reweighting loop: **proven insufficient** (plateaus at ~0.79; can't move the
  head without a generator). Discarded as the repro path.
- **Real single-GPU GRPO generator loop** (the faithful cheap version, `run_dpg_grpo_min.py`):
  launched on the **"67"** target — the run in flight. Plan: drive "67" to ~1.0 pixel accuracy
  (mechanism proof), then swap `Y` to the **21×21 QR** target and scale up the same loop for the
  headline result. **HF upload is held until a clean decoded pattern exists.**
