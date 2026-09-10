#!/usr/bin/env bash
# One sequential pipeline, in priority order. Replaces run_word_level_grid.sh
# and run_bert_seeds.sh, which were a grid plus a *watcher* that polled its log
# and killed it when done.
#
# That watcher destroyed the whole session. Two bugs, both avoided here by not
# having a watcher at all:
#   * its `pkill -f run_word_level_grid.sh` matched the tmux SERVER's own
#     command line -- tmux new-session records the command it was given -- so
#     it killed the server and every unrelated session on it;
#   * its 6h timeout fell through to "stop anyway" instead of aborting, so a
#     phase that was merely slower than estimated got torn down mid-run.
# Sequential phases need no cross-process signalling, so neither can recur.
#
#   tmux new-session -d -s iolex \
#     "bash scripts/run_pipeline.sh 2>&1 | tee outputs/pipeline.log"
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -Is)] === 1/5  backfill selection_log.npz into finished selector runs ==="
# Minutes each: no training, and runtime.eval.stsb=false skips the sweep whose
# artifact those runs already have.
.venv/bin/python scripts/backfill_selection_log.py --device cuda
rc=$?
echo "[$(date -Is)] BACKFILL_EXIT=$rc"

echo "[$(date -Is)] === 2/5  wikiann bert/mean, seeds 3,4,5 ==="
# Seeds 0-2 exist; resume is seed-scoped so these train from scratch.
.venv/bin/forge grid data.dataset=wikiann data.encoder.family=bert data.encoder.pooling=mean \
    runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 train.continue=true \
    --sweep runtime.seed=3,4,5
rc=$?
echo "[$(date -Is)] WIKIANN_EXIT=$rc"

echo "[$(date -Is)] === 3/5  conll2003 bert/mean, seeds 0..5 ==="
.venv/bin/forge grid data.dataset=conll2003 data.encoder.family=bert data.encoder.pooling=mean \
    runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 train.continue=true \
    --sweep runtime.seed=0,1,2,3,4,5
rc=$?
echo "[$(date -Is)] CONLL_EXIT=$rc"

echo "[$(date -Is)] === 4/5  the abstraction-spectrum datasets ==="
# Three label sets spanning H(label|word)/H(label): upos 0.08 (the word decides),
# deprel 0.26 (needs the sentence), GUM RST discourse 0.86 (needs the argument).
# All are balanced and tag-rich, so the grounding correlation is far better
# powered than on wikiann: 14/29/31 usable tags against 6, which takes the |r|
# needed for p<0.05 from 0.811 down to 0.53/0.37/0.36.
#
# Tagger first for each: it is the cheaper half, and it exercises the whole data
# path (build, tokenize, cache, encode), so a dataset that is broken fails here
# in minutes rather than after a selector has trained for twenty.
for DS in ud_upos ud_deprel ud_discourse; do
    echo "[$(date -Is)] --- $DS tagger ---"
    .venv/bin/python -m tagger "$DS" --family bert --seeds 0,1,2 --device cuda \
        --set runtime.data.batch_size=32
    echo "[$(date -Is)] TAGGER_${DS}_EXIT=$?"

    echo "[$(date -Is)] --- $DS selector ---"
    .venv/bin/forge grid data.dataset="$DS" data.encoder.family=bert data.encoder.pooling=mean \
        runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 train.continue=true \
        --sweep runtime.seed=0,1,2
    echo "[$(date -Is)] SELECTOR_${DS}_EXIT=$?"
done

echo "[$(date -Is)] === 5/5  remaining wikiann oracles (~3h each, longest pole, so last) ==="
# bert is already done. electra was killed mid-eval and re-runs from scratch:
# its search is not resumable from the masks it had written.
.venv/bin/forge grid task=oracle data.dataset=wikiann data.encoder.pooling=mean \
    runtime.device=cuda runtime.data.batch_size=128 \
    --sweep data.encoder.family=electra,roberta,sbert,e5
rc=$?
echo "[$(date -Is)] ORACLE_EXIT=$rc"
.venv/bin/forge run task=oracle data.dataset=wikiann data.encoder.family=llm \
    data.encoder.pooling=last runtime.device=cuda runtime.data.batch_size=128 \
    runtime.oracle.chunk_tokens=65536
rc=$?
echo "[$(date -Is)] ORACLE_LLM_EXIT=$rc"
echo "[$(date -Is)] === pipeline done ==="
