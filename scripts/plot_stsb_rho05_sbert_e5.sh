#!/usr/bin/env bash
# Runs on CPU (6 threads): the GPU is full -- another user's jobs plus
# entity_direction.py -- and pythia's CPU training keeps the other 18.
# 1) Retrain e5 seed 1 of the rho>=0.5 STS-B selectors (the first attempt died at
#    its first step, exception swallowed by forge). 2) Plot sbert + e5 rho>=0.5
#    selectors on STS-B: rows figure + span-start figure from one sweep.
set -u
cd /home/simonelusetti/Iolex

echo "[$(date -Is)] retraining e5 seed 1"
.venv/bin/forge --config-dir old/conf --config-name config -M train grid \
    data.dataset=stsb data.encoder.family=e5 data.encoder.pooling=mean \
    'model.loss.sweep_range=[0.5,1.0,6]' +model.selector.keep_nonword=true \
    runtime.device=cpu runtime.threads=6 runtime.grid=true train.continue=true runtime.data.batch_size=16 \
    --sweep runtime.seed=1
echo "[$(date -Is)] e5 seed-1 exit=$?"
.venv/bin/python -W ignore - <<'PY'
import sys; sys.path.insert(0, ".")
from utils.forge_paths import discover_strategies
for s in discover_strategies("stsb"):
    if s["label"].endswith("/train-rho0.5-1"):
        print(f"  {s['label']}: {len(s['runs'])} completed runs")
PY

echo "[$(date -Is)] evaluating sbert + e5 (rho>=0.5 selectors) on STS-B"
.venv/bin/python -W ignore analysis/scripts/stsb_both_sides.py \
    --figure both --train-rho 0.5-1 --families sbert,e5 --random-runs 3 --device cpu --threads 6 \
    --rhos 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    --trap-spans 0.1-1.0,0.2-1.0,0.3-1.0,0.4-1.0,0.5-1.0,0.6-1.0,0.7-1.0,0.8-1.0,0.9-1.0,0.1-0.5 \
    --row-spans 0.1-1.0,0.1-0.5,0.5-1.0
echo "[$(date -Is)] done"
