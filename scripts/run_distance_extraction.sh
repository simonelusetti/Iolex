#!/usr/bin/env bash
# Per-sentence reconstruction distance for every (dataset, encoder) with a selector.
#
#   tmux new-session -d -s dist \
#     "bash scripts/run_distance_extraction.sh 2>&1 | tee outputs/distance_extraction.log"
#
# 1 - cos(pool(suppressed), pool(full)) per sentence per rho -- the selector's own
# training objective, which nothing in the pipeline saves. Each run reloads the
# trained selector and re-runs the test split; the forward is evaluated at all
# nine budgets, so this is ~10 encoder passes per batch and the large corpora
# dominate. Validated per seed against the loss the forward returns.
#
# Coverage is uneven by design: llm and simcse selectors were only trained where
# we ran them. conll2003 is already cached for all three.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () {
    echo "[$(date -Is)] --- $1 : $2 ---"
    .venv/bin/python -W ignore analysis/scripts/sentence_distance.py "$1" --families "$2" --device cuda
    echo "[$(date -Is)] DIST_$1_EXIT=$?"
}
# smallest first, so partial results are useful early
run wikiann          sbert,simcse
run conll2000        sbert,llm,simcse
run ud_upos          sbert
run ud_deprel        sbert
run ud_discourse     sbert
run movie_rationales sbert,llm,simcse
run fewnerd          sbert

echo "[$(date -Is)] === caches on disk ==="
.venv/bin/python -W ignore -c "
import sys; sys.path.insert(0,'.')
from tagger.tagging import store_dir
for ds in ['conll2003','wikiann','fewnerd','ud_upos','ud_deprel','ud_discourse','conll2000','movie_rationales']:
    got=[f for f in ['sbert','llm','simcse']
         if (store_dir(ds,f,0).parent/'sentence_distance.npz').exists()]
    print(f'  {ds:18s} {got}')
" 2>&1 | grep -v skipping
echo "[$(date -Is)] done"
