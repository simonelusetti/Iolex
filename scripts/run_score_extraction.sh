#!/usr/bin/env bash
# Extract the selector's continuous score for every (dataset, encoder) that has
# both a trained selector and a probe, so p(true) can be plotted against it.
#
#   tmux new-session -d -s scores \
#     "bash scripts/run_score_extraction.sh 2>&1 | tee outputs/score_extraction.log"
#
# The score is never written to disk by the pipeline -- selection_log.npz keeps
# only the binary top-rho decision -- so each trained selector is reloaded and the
# test split re-run. See analysis/scripts/selector_scores.py.
#
# conll2000 and movie_rationales also need their per-word probe outputs built
# first (word_correct.npz), which nothing has computed yet.
#
# fewnerd_fine is omitted: it has probes but no trained selector for any of these
# three encoders. wikiann/llm is omitted: its only llm run is an ORACLE, which is
# a mask search with no selector to score.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -Is)] === 1/2  per-word probe outputs for the two new corpora ==="
for DS in conll2000 movie_rationales; do
    .venv/bin/python -W ignore -c "
import sys; sys.path.insert(0,'.')
from utils.word_auc import probe_words
for fam in ['sbert','e5','llm']:
    d, why = probe_words('$DS', fam, 'cuda')
    print(f'  $DS/{fam}: ' + ('ok' if d is not None else why), flush=True)
"
    echo "[$(date -Is)] PROBEWORDS_${DS}_EXIT=$?"
done

echo "[$(date -Is)] === 2/2  selector scores ==="
run () {  # dataset, families
    echo "[$(date -Is)] --- $1 : $2 ---"
    .venv/bin/python -W ignore analysis/scripts/selector_scores.py "$1" --families "$2" --device cuda
    echo "[$(date -Is)] SCORES_$1_EXIT=$?"
}
run wikiann          sbert,e5
run fewnerd          sbert,e5
run ud_upos          sbert,e5
run ud_deprel        sbert,e5
run ud_discourse     sbert,e5
run conll2000        sbert,e5,llm
run movie_rationales sbert,e5,llm

echo "[$(date -Is)] === what exists now ==="
.venv/bin/python -W ignore -c "
import sys; sys.path.insert(0,'.')
from tagger.tagging import store_dir
from pathlib import Path
for ds in ['conll2003','wikiann','fewnerd','ud_upos','ud_deprel','ud_discourse','conll2000','movie_rationales']:
    got=[f for f in ['sbert','e5','llm']
         if (store_dir(ds,f,0).parent/'selector_scores.npz').exists()]
    print(f'  {ds:18s} {got}')
"
echo "[$(date -Is)] done"
