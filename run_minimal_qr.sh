#!/usr/bin/env bash
# Minimal end-to-end QR/"67" metagradient reproduction (arXiv 2604.08423).
# Runs the authors' JAX metagrad engine on GPT-2 with the sixseven target on ONE GPU,
# closing the loop via metagradient ascent on per-example data weights (no verl/8-GPU stack).
# Emits EVAL.md (repo root) + decoded-image artifacts to .openresearch/artifacts/.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOCAL_FAST_STORAGE="${LOCAL_FAST_STORAGE:-/tmp/minimal_qr}"
mkdir -p "$LOCAL_FAST_STORAGE"
mkdir -p "$REPO_DIR/.openresearch/artifacts"

echo "=== minimal-qr repro ==="
echo "repo:    $REPO_DIR"
echo "scratch: $LOCAL_FAST_STORAGE"
nvidia-smi -L || true

# uv builds the dataset_metagradients_jax env from pyproject.toml + uv.lock on first run.
cd "$REPO_DIR/dataset_metagradients_jax"

# Run from repo root so EVAL.md + .openresearch/artifacts/ land at the root (synced by orx).
ARTIFACTS_DIR="$REPO_DIR/.openresearch/artifacts"
uv run python -u scripts/run_minimal_qr.py \
    --outer-steps "${OUTER_STEPS:-40}" \
    --inner-steps "${INNER_STEPS:-96}" \
    --pool-size "${POOL_SIZE:-256}" \
    --microbatch-size "${MICROBATCH_SIZE:-32}" \
    --lr-inner "${LR_INNER:-1.0e-3}" \
    --alpha-w "${ALPHA_W:-4.0}" \
    --max-w "${MAX_W:-64.0}" \
    --artifacts "$ARTIFACTS_DIR"

# The driver writes EVAL.md into its CWD (dataset_metagradients_jax/); move it to repo root.
if [ -f "EVAL.md" ]; then mv -f EVAL.md "$REPO_DIR/EVAL.md"; fi
echo "=== minimal-qr repro done; EVAL.md at $REPO_DIR/EVAL.md ==="
