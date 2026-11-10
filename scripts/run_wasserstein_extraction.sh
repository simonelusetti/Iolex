#!/usr/bin/env bash
# Surviving-word Wasserstein distance (before vs after re-encoding), first selector
# seed, for the angle plot (analysis/scripts/distance_by_a_all.py).
#
#   tmux new-session -d -s wass \
#     "bash scripts/run_wasserstein_extraction.sh 2>&1 | tee outputs/wasserstein.log"
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run () {   # dataset families [batch]
    echo "[$(date -Is)] --- $1 / $2 ---"
    .venv/bin/python -W ignore analysis/scripts/sentence_distance.py "$1" --families "$2" \
        --device cuda --wasserstein ${3:+--batch-size $3} 2>&1 | grep -v "INFO\|it/s"
}
run conll2003        e5,llm,simcse
run wikiann          sbert,e5,llm,simcse
run conll2000        sbert,e5,llm,simcse
run movie_rationales sbert,e5,simcse 4
run movie_rationales llm 2
echo "[$(date -Is)] done"
