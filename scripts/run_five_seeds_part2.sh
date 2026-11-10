#!/usr/bin/env bash
# Continuation of run_five_seeds.sh, with e5 retrained under the prefix fix.
#
#   tmux new-session -d -s fiveseeds2 \
#     "bash scripts/run_five_seeds_part2.sh 2>&1 | tee outputs/five_seeds_part2.log"
#
# run_five_seeds.sh was stopped after conll2003 (sbert, e5, simcse, llm seeds 3,4)
# because the remaining e5 entries would have trained more selectors with the
# "query: " prefix masked out. It first waits for that job's last forge process
# (conll2003 llm seeds 3,4) to finish -- forge run ids are random uuids, so a
# killed run cannot be resumed, only restarted.
#
# e5: all five seeds, fresh, with +model.selector.keep_nonword=true, which keeps
# the prefix in attention at every rho and gives the runs their own experiment
# signature. The old e5 runs stay in the store under their old signature and are
# labelled "e5/mean/prefix-masked" (utils/forge_paths.variant_suffix); their
# conll2003 seeds 3,4 already ran and are kept with them.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WAIT_PID=${WAIT_PID:-}
if [ -n "$WAIT_PID" ]; then
    echo "[$(date -Is)] waiting for pid $WAIT_PID (conll2003 llm seeds 3,4) ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date -Is)] pid $WAIT_PID finished"
fi

sel () {   # dataset family pooling batch seeds [extra overrides...]
    local DS=$1 FAM=$2 POOL=$3 BS=$4 SEEDS=$5; shift 5
    echo "[$(date -Is)] --- $DS / $FAM / $POOL  seeds=$SEEDS $* ---"
    .venv/bin/forge --config-dir old/conf --config-name config -M train grid \
        data.dataset="$DS" data.encoder.family="$FAM" data.encoder.pooling="$POOL" \
        runtime.device=cuda runtime.grid=true runtime.data.batch_size="$BS" \
        train.continue=true "$@" \
        --sweep runtime.seed="$SEEDS"
    echo "[$(date -Is)] EXIT_${DS}_${FAM}=$?"
}

echo "[$(date -Is)] === 1/2  e5 with the prefix kept, seeds 0-4 ==="
sel conll2003        e5 mean 16 0,1,2,3,4 +model.selector.keep_nonword=true
sel conll2000        e5 mean  8 0,1,2,3,4 +model.selector.keep_nonword=true
sel movie_rationales e5 mean  8 0,1,2,3,4 +model.selector.keep_nonword=true
sel wikiann          e5 mean 16 0,1,2,3,4 +model.selector.keep_nonword=true

echo "[$(date -Is)] === 2/2  the rest of the five-seed top-up ==="
sel wikiann sbert  mean 16 3,4
sel wikiann simcse cls  16 3,4 model.keep_special=true
sel wikiann llm    last  8 3,4
for DS in conll2000 movie_rationales; do
    sel "$DS" sbert  mean  8 0,1,2,3
    sel "$DS" simcse cls  16 3,4 model.keep_special=true
    sel "$DS" llm    last  8 0,1,2,3
done

echo "[$(date -Is)] === what landed ==="
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
from utils.plot_grounding import discover_series
for ds in ("conll2003", "wikiann", "conll2000", "movie_rationales"):
    for s in discover_series(ds):
        if "oracle" not in s["label"] and s["family"] in ("sbert", "e5", "llm", "simcse"):
            print(f"  {ds:18s} {s['label']:24s} runs={len(s['bias_runs'])}")
PY
echo "[$(date -Is)] done"
