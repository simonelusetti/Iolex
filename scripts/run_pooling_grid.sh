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

# Overridable so the same verified path can backfill a corpus that was
# skipped the first time round:  DATASETS=wikiann bash scripts/run_pooling_grid.sh
DATASETS="${DATASETS:-conll2003 ud_upos ud_deprel ud_discourse}"
SEEDS="${SEEDS:-0,1,2}"

echo "[$(date -Is)] selector seeds=$SEEDS"

echo "[$(date -Is)] === selectors: bert min/max, seeds $SEEDS ==="
for DS in $DATASETS; do
    echo "[$(date -Is)] --- $DS selector min/max ---"
    .venv/bin/forge grid data.dataset="$DS" data.encoder.family=bert \
        runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 \
        train.continue=true \
        --sweep data.encoder.pooling=min,max --sweep runtime.seed="$SEEDS"
    rc=$?
    echo "[$(date -Is)] SELECTOR_${DS}_EXIT=$rc"
done

# The oracle phase used to live here and has been removed. Measured cost was
# ~43h per pooling at the configured caps (conll2003 180M candidate masks,
# ud_discourse 114M, ud_upos and ud_deprel 84M each), against ~2h for all 24
# selector runs. It was cancelled mid-conll2003; nothing partial survives, the
# run was purged. Re-add with `forge grid task=oracle ...` when the ceiling is
# actually needed -- the grounding correlation does not use it.

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
