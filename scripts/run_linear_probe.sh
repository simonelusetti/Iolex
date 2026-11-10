#!/usr/bin/env bash
# Linear NER probe (no hidden layer) on conll2003 for sbert, e5, pythia; 3 seeds.
#
#   tmux new-session -d -s linprobe \
#     "bash scripts/run_linear_probe.sh 2>&1 | tee outputs/linear_probe.log"
#
# Same data, optimiser, epochs (10) and batch size (32) as the MLP probes; only
# the head changes (tagger.hidden=0). Stored as outputs/probe/conll2003/<family>+linear.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for FAM in sbert e5 llm; do
    echo "[$(date -Is)] --- conll2003 / $FAM / linear ---"
    .venv/bin/python -m tagger conll2003 --family "$FAM" --seeds 0,1,2 --device cuda \
        --linear --set runtime.data.batch_size=32
    echo "[$(date -Is)] EXIT_${FAM}=$?"
done
echo "[$(date -Is)] done"
