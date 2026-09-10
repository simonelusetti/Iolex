#!/usr/bin/env bash
# Few-NERD: the third NER corpus, added to break the wikiann/conll2003 tie.
#
#   tmux new-session -d -s fewnerd \
#     "bash scripts/run_fewnerd.sh 2>&1 | tee outputs/fewnerd.log"
#
# The live question: on division/absolute, conll2003 (+0.51), ud_deprel (+0.52),
# ud_discourse (+0.38) and ud_upos (+0.34) agree, and wikiann alone says -0.50.
# Two NER corpora disagreeing cannot say whether that is a wikiann artifact or
# the NER task reasserting itself. A third can.
#
# Both granularities, because they answer different halves:
#   fewnerd       9 tags, the direct analogue of wikiann (7) and conll2003 (9)
#   fewnerd_fine 67 tags, critical |r| ~0.24 instead of ~0.65 -- powered enough
#                to be read on its own rather than only as a vote
#
# Tagger first: it is the cheap half and it exercises the whole data path, so a
# broken corpus fails in minutes rather than after a selector has trained.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS="fewnerd fewnerd_fine"

for DS in $DATASETS; do
    echo "[$(date -Is)] --- $DS tagger ---"
    .venv/bin/python -m tagger "$DS" --family bert --seeds 0,1,2 --device cuda \
        --set runtime.data.batch_size=32
    echo "[$(date -Is)] TAGGER_${DS}_EXIT=$?"
done

for DS in $DATASETS; do
    echo "[$(date -Is)] --- $DS selector bert/mean, seeds 0,1,2 ---"
    .venv/bin/forge grid data.dataset="$DS" data.encoder.family=bert \
        data.encoder.pooling=mean \
        runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 \
        train.continue=true \
        --sweep runtime.seed=0,1,2
    echo "[$(date -Is)] SELECTOR_${DS}_EXIT=$?"
done

# forge grid marks a failed entry and still exits 0, so ask the store instead.
echo "[$(date -Is)] === what exists now ==="
.venv/bin/python -c "
from utils.plot_grounding import discover_series
for ds in '$DATASETS'.split():
    s = discover_series(ds)
    print(f'{ds:14s}', ', '.join(f\"{x['label']}({len(x['bias_runs'])})\" for x in s) or 'none')
"
echo "[$(date -Is)] done"
