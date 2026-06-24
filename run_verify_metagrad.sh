#!/usr/bin/env bash
# Quick verification of the DPG core claim: the per-example metagradient is correct
# (incl. Theorem 3.1). Runs on the small from-scratch model, single GPU, minutes.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOCAL_FAST_STORAGE="${LOCAL_FAST_STORAGE:-/tmp/verify_metagrad}"
mkdir -p "$LOCAL_FAST_STORAGE" "$REPO_DIR/.openresearch/artifacts"
export ARTIFACTS_DIR="$REPO_DIR/.openresearch/artifacts"
echo "=== metagrad correctness verification ==="
nvidia-smi -L || true
cd "$REPO_DIR/dataset_metagradients_jax"
rc=0
uv run python -u scripts/verify_metagrad.py || rc=$?
if [ -f "EVAL.md" ]; then mv -f EVAL.md "$REPO_DIR/EVAL.md"; fi
echo "=== done (rc=$rc); EVAL.md at $REPO_DIR/EVAL.md ==="
exit $rc
