#!/usr/bin/env bash
# Retry of the E5 tag baselines (entities, POS) with small document batches
# after OOM on the shared GPU; then redraw both baseline figures.
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for attempt in 1 2 3 4 5; do
    echo "[$(date -Is)] e5 (attempt $attempt)"
    .venv/bin/python -W ignore analysis/scripts/home_task_baselines.py --task e5 --only entities,pos \
        --out-suffix _tags --doc-bs 16 && break
    echo "[$(date -Is)] FAILED e5 attempt $attempt"; sleep 600
done
.venv/bin/python -W ignore analysis/scripts/home_task_baselines_split_plot.py
echo "[$(date -Is)] TAGS_E5_DONE"
