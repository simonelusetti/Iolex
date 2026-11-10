#!/usr/bin/env bash
# STS-B-trained selectors for sbert, e5 (mean) and pythia (last), 3 seeds each.
#
#   tmux new-session -d -s stsb "bash scripts/run_stsb_selectors.sh 2>&1 | tee outputs/stsb_selectors.log"
#
# utils/grid_stsb.yaml left llm out because gpt_token_embeddings used to detach the
# mask; that was rewritten (tests/test_gpt_mask.py), so pythia trains now. pooling
# follows each encoder's convention: mean for the BERT stacks, last for pythia.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sel () {
    echo "[$(date -Is)] --- stsb / $1 / $2 ---"
    .venv/bin/forge --config-dir old/conf --config-name config -M train grid \
        data.dataset=stsb data.encoder.family="$1" data.encoder.pooling="$2" \
        runtime.device=cuda runtime.grid=true runtime.data.batch_size="$3" \
        train.continue=true --sweep runtime.seed=0,1,2
    echo "[$(date -Is)] EXIT_$1=$?"
}
sel sbert mean 16
sel e5    mean 16
sel llm   last 8
echo "[$(date -Is)] done"
