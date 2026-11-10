#!/usr/bin/env bash
# ue5T's two tag-based baselines (named entities first; POS ranking) on the three
# home tasks, after the stopword/TF-IDF run; merged at plot time, then both
# baseline figures are redrawn.
set -u
cd /home/simonelusetti/Iolex
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
until grep -q LEXICAL_DONE outputs/lexical.log 2>/dev/null; do sleep 60; done
for t in sbert e5 pythia; do
    for attempt in 1 2 3; do
        until [ "$(free_mib)" -ge 3000 ]; do sleep 120; done
        echo "[$(date -Is)] $t (attempt $attempt)"
        .venv/bin/python -W ignore analysis/scripts/home_task_baselines.py --task $t \
            --only entities,pos --out-suffix _tags && break
        echo "[$(date -Is)] FAILED $t attempt $attempt"; sleep 300
    done
done
.venv/bin/python -W ignore analysis/scripts/home_task_baselines_split_plot.py
echo "[$(date -Is)] TAGS_DONE"
