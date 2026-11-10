#!/usr/bin/env bash
# Home tasks scored with the selector's training-time soft mask (selector_soft),
# 3 seeds per encoder; the other curves come from the existing baseline runs.
set -u
cd /home/simonelusetti/Iolex
PY=".venv/bin/python -W ignore"
for t in sbert e5 pythia; do
    echo "[$(date -Is)] $t"
    $PY analysis/scripts/home_task_baselines.py --task $t --only selector_soft --out-suffix _soft \
        || { echo "[$(date -Is)] SOFT_FAILED $t"; exit 1; }
done
$PY analysis/scripts/home_task_softmask_plot.py || { echo "[$(date -Is)] SOFT_FAILED plot"; exit 1; }
echo "[$(date -Is)] SOFT_DONE"
