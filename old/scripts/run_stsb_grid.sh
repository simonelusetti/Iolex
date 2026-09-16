#!/usr/bin/env bash
# Train selectors ON STS-B (5 encoders x 3 seeds), then redraw the 3-panel figure.
#
#   tmux new-session -d -s stsb \
#     "bash scripts/run_stsb_grid.sh 2>&1 | tee outputs/stsb_grid.log"
#
# This fills panels 1 and 2 of the figure, which need data.dataset=stsb and had
# no runs at all. Panel 3 (selectors trained on the tagging corpora, evaluated on
# STS-B) already has 92 runs, because runtime.eval.stsb defaults to true and every
# training run is STS-B-evaluated at the end.
#
# llm/pythia is excluded from the grid -- its selector cannot receive gradient
# (see the comment in utils/grid_stsb.yaml), so it would produce an untrained
# curve. STS-B is small (5.7k train pairs), so this shares the GPU with the
# alignment grid rather than waiting on it.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -Is)] === training selectors on STS-B ==="
.venv/bin/forge grid --file utils/grid_stsb.yaml
echo "[$(date -Is)] STSB_GRID_EXIT=$?"

echo "[$(date -Is)] === redrawing the figure ==="
.venv/bin/python -W ignore utils/plot_stsb_sufficiency.py
echo "[$(date -Is)] PLOT_EXIT=$?"
echo "[$(date -Is)] done"
