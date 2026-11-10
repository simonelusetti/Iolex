#!/usr/bin/env bash
# Pythia selector on wikiann (3 seeds), then its per-sentence reconstruction distances.
#
#   tmux new-session -d -s pythiawiki \
#     "bash scripts/run_pythia_wikiann.sh 2>&1 | tee outputs/pythia_wikiann.log"
#
# wikiann previously had only an ORACLE for pythia (llm/last/oracle) -- a mask
# search, not a trained selector -- so it could not enter any selector analysis.
# The probe already exists (3 seeds, with weights), so only the selector is needed.
#
# The family key stays `llm`: it is part of the experiment signature, and renaming
# it would orphan every existing pythia run. "pythia" is the display name only.
#
# Settings match the other pythia selectors: pooling=last (causal attention means
# only the final token has seen the whole sequence), keep_special=false, and
# batch_size=8 because the differentiable mask forces SDPA onto its MATH kernel.
# pythia has no attended non-word positions, so the cls-only mask exemption in
# RationaleSelectorModel does not apply to it.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -Is)] === 1/2  pythia selector on wikiann, seeds 0,1,2 ==="
.venv/bin/forge --config-dir old/conf --config-name config -M train grid \
    data.dataset=wikiann data.encoder.family=llm data.encoder.pooling=last \
    runtime.device=cuda runtime.grid=true runtime.data.batch_size=8 \
    train.continue=true \
    --sweep runtime.seed=0,1,2
echo "[$(date -Is)] SELECTOR_EXIT=$?"

echo "[$(date -Is)] === 2/2  reconstruction distances ==="
.venv/bin/python -W ignore analysis/scripts/sentence_distance.py wikiann --families llm --device cuda
echo "[$(date -Is)] DIST_EXIT=$?"
echo "[$(date -Is)] done"
