#!/usr/bin/env bash
# Single-GPU faithful DPG QR/"67" reproduction: small JAX GRPO generator + the authors'
# real metagradient reward (GPT-2 sixseven target). Emits EVAL.md + artifacts.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOCAL_FAST_STORAGE="${LOCAL_FAST_STORAGE:-/tmp/dpg_grpo_min}"
mkdir -p "$LOCAL_FAST_STORAGE" "$REPO_DIR/.openresearch/artifacts"
export METAGRAD_WANDB_NAME="${METAGRAD_WANDB_NAME:-dpg-grpo-min}"

echo "=== single-GPU DPG GRPO repro ==="
nvidia-smi -L || true
cd "$REPO_DIR/dataset_metagradients_jax"

# Target model to encode the pattern into (default Qwen3-0.6B-Base; the metric is robust to
# tied embeddings). Smoke config: T=64 real inner Adam steps, M=20 GRPO steps -- cheap, and
# enough to PROVE the fix (Phi must leave -ln2 and pixel_acc must respond) before scaling M up.
export TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-0.6B-Base}"
uv run python -u scripts/run_dpg_grpo_min.py \
    --grpo-steps "${GRPO_STEPS:-6}" \
    --inner-steps "${INNER_STEPS:-32}" \
    --n-prompts "${N_PROMPTS:-4}" \
    --group-size "${GROUP_SIZE:-8}" \
    --microbatch-size "${MICROBATCH_SIZE:-4}" \
    --lr-inner "${LR_INNER:-2.0e-3}" \
    --lr-gen "${LR_GEN:-1.0e-4}" \
    --artifacts "$REPO_DIR/.openresearch/artifacts"

if [ -f "EVAL.md" ]; then mv -f EVAL.md "$REPO_DIR/EVAL.md"; fi
echo "=== done; EVAL.md at $REPO_DIR/EVAL.md ==="
