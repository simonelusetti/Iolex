#!/usr/bin/env bash
# bert min/max on every dataset that currently has only bert/mean.
#
#   tmux new-session -d -s pooling \
#     "bash scripts/run_pooling_grid.sh 2>&1 | tee outputs/pooling_grid.log"
#
# Why this exists: the correlation summary heatmap has exactly one populated
# column (wikiann) because every other corpus has a single bert/mean series and
# nothing to compare it against. min and max are the pooling axis the paper
# varies, so filling them in is what makes those columns readable.
#
# No tagger phase: probes are keyed by (dataset, family) and never pool, so the
# bert probes already cached on all four datasets serve min and max unchanged.
#
# Sequential on purpose -- one GPU, and a watcher process is what destroyed the
# tmux server once already (see run_pipeline.sh).
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS="conll2003 ud_upos ud_deprel ud_discourse"
SEEDS="${SEEDS:-0,1,2}"
ORACLE_SEEDS="${ORACLE_SEEDS:-}"     # empty = one unseeded oracle per cell

echo "[$(date -Is)] selector seeds=$SEEDS  oracle seeds=${ORACLE_SEEDS:-<single>}"

echo "[$(date -Is)] === 1/2  selectors: bert min/max, seeds $SEEDS ==="
for DS in $DATASETS; do
    echo "[$(date -Is)] --- $DS selector min/max ---"
    .venv/bin/forge grid data.dataset="$DS" data.encoder.family=bert \
        runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 \
        train.continue=true \
        --sweep data.encoder.pooling=min,max --sweep runtime.seed="$SEEDS"
    echo "[$(date -Is)] SELECTOR_${DS}_EXIT=$?"
done

# Oracles last: they are the long pole by an order of magnitude. Measured
# candidate counts at the configured caps, and ~3k masks/s:
#   conll2003 180M (~17h)   ud_upos 84M (~8h)
#   ud_deprel  84M (~8h)    ud_discourse 114M (~11h)
# That is ~43h per pooling per seed, and pooling does not change the count --
# only what each candidate is scored with.
echo "[$(date -Is)] === 2/2  oracles: bert min/max ==="
for DS in $DATASETS; do
    echo "[$(date -Is)] --- $DS oracle min/max ---"
    if [ -n "$ORACLE_SEEDS" ]; then
        .venv/bin/forge grid task=oracle data.dataset="$DS" data.encoder.family=bert \
            runtime.device=cuda runtime.data.batch_size=128 \
            --sweep data.encoder.pooling=min,max --sweep runtime.seed="$ORACLE_SEEDS"
    else
        .venv/bin/forge grid task=oracle data.dataset="$DS" data.encoder.family=bert \
            runtime.device=cuda runtime.data.batch_size=128 \
            --sweep data.encoder.pooling=min,max
    fi
    echo "[$(date -Is)] ORACLE_${DS}_EXIT=$?"
done

# forge grid marks a failed entry and still exits 0, so an exit code above
# proves nothing. Ask the store what actually finished.
echo "[$(date -Is)] === what exists now ==="
.venv/bin/python -c "
from utils.plot_grounding import discover_series
for ds in '$DATASETS'.split():
    s = discover_series(ds)
    print(f'{ds:14s}', ', '.join(f\"{x['label']}({len(x['bias_runs'])})\" for x in s) or 'none')
"
echo "[$(date -Is)] done"
