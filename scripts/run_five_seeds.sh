#!/usr/bin/env bash
# Top every (dataset, encoder) selector up to 5 seeds.
#
#   tmux new-session -d -s fiveseeds \
#     "bash scripts/run_five_seeds.sh 2>&1 | tee outputs/five_seeds.log"
#
# What existed (2026-09-18):
#   conll2003, wikiann   sbert, e5, llm, simcse     seeds 0,1,2   -> add 3,4
#   conll2000, movie     simcse                     seeds 0,1,2   -> add 3,4
#   conll2000, movie     sbert, e5, llm             seed 42 only  -> add 0,1,2,3
#
# Settings copy the existing runs of each pair, so the new seeds land in the same
# experiment signature and only runtime.seed differs:
#   sbert/e5 pooling=mean, llm pooling=last, simcse pooling=cls + keep_special=true;
#   batch 16 for sbert/e5/simcse on conll2003/wikiann and for simcse everywhere,
#   batch 8 for llm everywhere and for sbert/e5 on conll2000/movie (as they ran).
# Config is old/conf/config.yaml (the root one is the NER tagger's).
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sel () {   # dataset family pooling batch seeds [extra overrides...]
    local DS=$1 FAM=$2 POOL=$3 BS=$4 SEEDS=$5; shift 5
    echo "[$(date -Is)] --- $DS / $FAM / $POOL  seeds=$SEEDS ---"
    .venv/bin/forge --config-dir old/conf --config-name config -M train grid \
        data.dataset="$DS" data.encoder.family="$FAM" data.encoder.pooling="$POOL" \
        runtime.device=cuda runtime.grid=true runtime.data.batch_size="$BS" \
        train.continue=true "$@" \
        --sweep runtime.seed="$SEEDS"
    echo "[$(date -Is)] EXIT_${DS}_${FAM}=$?"
}

for DS in conll2003 wikiann; do
    sel "$DS" sbert  mean 16 3,4
    sel "$DS" e5     mean 16 3,4
    sel "$DS" simcse cls  16 3,4 model.keep_special=true
    sel "$DS" llm    last  8 3,4
done
for DS in conll2000 movie_rationales; do
    sel "$DS" sbert  mean  8 0,1,2,3
    sel "$DS" e5     mean  8 0,1,2,3
    sel "$DS" simcse cls  16 3,4 model.keep_special=true
    sel "$DS" llm    last  8 0,1,2,3
done

echo "[$(date -Is)] === what landed ==="
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
from utils.plot_grounding import discover_series
for ds in ("conll2003", "wikiann", "conll2000", "movie_rationales"):
    for s in discover_series(ds):
        if "oracle" not in s["label"] and s["label"].split("/")[0] in ("sbert", "e5", "llm", "simcse"):
            print(f"  {ds:18s} {s['label']:14s} runs={len(s['bias_runs'])}")
PY
echo "[$(date -Is)] done"
