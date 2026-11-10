#!/usr/bin/env bash
# PrunePert baseline on the home tasks: train 3 scorer seeds per encoder
# (analysis/scripts/prunepert.py), evaluate at input level with the other
# baselines' protocol, then draw one figure per encoder.
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/python -W ignore"
for t in sbert e5 pythia; do
    for s in 0 1 2; do
        [ -s outputs/prunepert/${t}_seed$s.pt ] && { echo "skip $t seed $s (done)"; continue; }
        $PY analysis/scripts/prunepert.py --task $t --seed $s || { echo "[$(date -Is)] PRUNEPERT_FAILED train $t $s"; exit 1; }
    done
done
for t in sbert e5 pythia; do
    echo "[$(date -Is)] evaluating $t"
    $PY analysis/scripts/home_task_baselines.py --task $t --only prunepert --out-suffix _prunepert \
        || { echo "[$(date -Is)] PRUNEPERT_FAILED eval $t"; exit 1; }
done
$PY analysis/scripts/home_task_prunepert_plot.py || { echo "[$(date -Is)] PRUNEPERT_FAILED plot"; exit 1; }
echo "[$(date -Is)] PRUNEPERT_DONE"
