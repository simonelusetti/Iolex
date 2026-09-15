#!/usr/bin/env bash
# The encoder axis on every corpus that lacks it.
#
#   tmux new-session -d -s encaxis \
#     "bash scripts/run_encoder_axis.sh 2>&1 | tee outputs/encoder_axis.log"
#
# Why: every encoder claim in the paper currently rests on wikiann alone, and
# wikiann is the corpus that flips sign against the other six on the grounding
# correlation (-0.50 where conll2003, ud_deprel, ud_discourse and ud_upos run
# +0.34 to +0.52). An encoder ordering measured only there could be a property
# of that corpus rather than of the encoders. This replicates it elsewhere.
#
# llm is excluded deliberately: its selector has never trained. HuggingFace's
# causal model does not propagate gradients through attention_mask, and `last`
# pooling passes none through the mask either, so the selector gets no signal.
# That needs a code decision (switch llm to mean, or make gpt_token_embeddings
# differentiable), not more compute. llm exists as an oracle on wikiann only.
#
# Tagger before selector, per dataset: it is the cheap half and it exercises
# the whole data path -- tokenizer group, cache build, encode -- so a broken
# (dataset, family) pair fails in minutes instead of after a selector has
# trained for twenty. Only conll2003 has non-bert tokenizer caches already;
# the ud_* and fewnerd* ones get built on first touch, which is CPU work
# inside the first tagger call for each family.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS="${DATASETS:-conll2003 ud_upos ud_deprel ud_discourse fewnerd fewnerd_fine}"
FAMILIES="${FAMILIES:-electra roberta sbert e5}"
SEEDS="${SEEDS:-0,1,2}"

echo "[$(date -Is)] datasets: $DATASETS"
echo "[$(date -Is)] families: $FAMILIES   seeds: $SEEDS"

for DS in $DATASETS; do
    for FAM in $FAMILIES; do
        echo "[$(date -Is)] --- $DS / $FAM tagger ---"
        .venv/bin/python -m tagger "$DS" --family "$FAM" --seeds "$SEEDS" --device cuda \
            --set runtime.data.batch_size=32
        rc=$?
        echo "[$(date -Is)] TAGGER_${DS}_${FAM}_EXIT=$rc"
    done

    echo "[$(date -Is)] --- $DS selectors: $FAMILIES ---"
    .venv/bin/forge grid data.dataset="$DS" data.encoder.pooling=mean \
        runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 \
        train.continue=true \
        --sweep data.encoder.family="$(echo "$FAMILIES" | tr ' ' ',')" \
        --sweep runtime.seed="$SEEDS"
    rc=$?
    echo "[$(date -Is)] SELECTOR_${DS}_EXIT=$rc"
done

# forge grid marks a failed entry and still exits 0, and the tagger loop above
# only reports its own status -- so neither proves anything. Ask the store.
echo "[$(date -Is)] === what exists now ==="
.venv/bin/python -c "
import tagger
from utils.plot_grounding import discover_series
fams = '$FAMILIES'.split()
for ds in '$DATASETS'.split():
    got = {s['label']: len(s['bias_runs']) for s in discover_series(ds)}
    sel = ', '.join(f'{f}({got.get(f+\"/mean\", 0)})' for f in fams)
    probes = ', '.join(f'{f}({len(tagger.load(ds, f))})' for f in fams)
    print(f'{ds:14s} selectors: {sel:52s} probes: {probes}')
"
echo "[$(date -Is)] done"
