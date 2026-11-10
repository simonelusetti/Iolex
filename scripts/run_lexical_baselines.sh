#!/usr/bin/env bash
# Stopword removal and TF-IDF as separate baselines on the three home tasks
# (cheap, evaluation only); merged with baselines_<task>.json at plot time.
set -u
cd /home/simonelusetti/Iolex
for t in sbert e5 pythia; do
    echo "[$(date -Is)] $t"
    .venv/bin/python -W ignore analysis/scripts/home_task_baselines.py --task $t \
        --only stopwords,tfidf --out-suffix _lexical || echo "[$(date -Is)] FAILED $t"
done
echo "[$(date -Is)] LEXICAL_DONE"
