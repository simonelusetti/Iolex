#!/usr/bin/env bash
# The selector vs stronger baselines (IDF, occlusion, LLMLingua-2, greedy oracle)
# on each encoder's home task: SBERT/STS-B, E5/BEIR, Pythia/Pile.
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for task in sbert e5 pythia; do
    echo "[$(date -Is)] --- $task ---"
    .venv/bin/python -W ignore analysis/scripts/home_task_baselines.py --task $task --oracle-limit 200 2>&1 \
        | grep --line-buffered -vE "it/s|Generating|Downloading|Fetching"
    echo "[$(date -Is)] $task exit=${PIPESTATUS[0]}"
done
echo "[$(date -Is)] done"
