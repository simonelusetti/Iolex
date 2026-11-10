#!/usr/bin/env bash
# Regenerates every STS-B both-sides / trapezoid figure with word-level scoring
# (src/retrival_fun.py now passes word_ids; rho counts words, as in training).
# Settings of each figure are those stored in its previous .npz. The previous,
# piece-scored .npz/.png are copied to analysis/piece_scored/ first.
set -u
cd /home/simonelusetti/Iolex
mkdir -p analysis/piece_scored
cp -n analysis/stsb_both_sides*.{npz,png} analysis/stsb_trapezoid_span_start*.{npz,png} \
      analysis/stsb_trapezoid_best_rho_*.png analysis/piece_scored/
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
cpuf () { if [ "$(free_mib)" -ge 12000 ]; then echo ""; else echo "--cpu-families llm"; fi; }
PY=".venv/bin/python -W ignore analysis/scripts/stsb_both_sides.py --random-runs 3 --threads 18"
R10="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
R5="0.5,0.6,0.7,0.8,0.9,1.0"
FROM01="0.1-1.0,0.2-1.0,0.3-1.0,0.4-1.0,0.5-1.0,0.6-1.0,0.7-1.0,0.8-1.0,0.9-1.0"
FROM05="0.5-1.0,0.6-1.0,0.7-1.0,0.8-1.0,0.9-1.0"
step () { echo "[$(date -Is)] $*"; }

step "1/8 stsb_both_sides (rows, stsb selectors)"
$PY --figure rows --families sbert,e5,llm $(cpuf) --rhos $R10 --trap-spans 0.1-1.0,0.1-0.5,0.5-1.0
step "2/8 stsb_both_sides_selectors_conll2003 (rows)"
$PY --figure rows --selector-dataset conll2003 --families sbert,e5,llm $(cpuf) --rhos $R10 \
    --trap-spans 0.1-1.0,0.1-0.5,0.5-1.0
step "3/8 stsb_trapezoid_span_start (stsb selectors)"
$PY --figure spans --families sbert,e5,llm $(cpuf) --rhos $R10 --trap-spans $FROM01
step "4/8 train-rho0.5-1, sbert+e5, from 0.1: rows + spans (spans renamed _sbert-e5_from0.1)"
$PY --figure both --train-rho 0.5-1 --families sbert,e5 --rhos $R10 \
    --trap-spans $FROM01,0.1-0.5 --row-spans 0.1-1.0,0.1-0.5,0.5-1.0
for ext in npz png; do
    mv analysis/stsb_trapezoid_span_start_train-rho0.5-1.$ext \
       analysis/stsb_trapezoid_span_start_train-rho0.5-1_sbert-e5_from0.1.$ext
done
step "5/8 stsb_trapezoid_span_start_train-rho0.5-1 (from 0.5)"
$PY --figure spans --train-rho 0.5-1 --families sbert,e5,llm $(cpuf) --rhos $R5 --trap-spans $FROM05
step "6/8 span start, wikiann selectors"
$PY --figure spans --selector-dataset wikiann --train-rho 0.5-1 --families sbert,e5,llm $(cpuf) \
    --rhos $R5 --trap-spans $FROM05
step "7/8 span start, conll2003 selectors"
$PY --figure spans --selector-dataset conll2003 --train-rho 0.5-1 --families sbert,e5,llm $(cpuf) \
    --rhos $R5 --trap-spans $FROM05
step "8/8 best single rho, conll2003 + wikiann"
.venv/bin/python -W ignore analysis/scripts/trapezoid_best_rho.py --sources conll2003,wikiann --variant train-rho0.5-1
.venv/bin/python -W ignore analysis/scripts/trapezoid_best_rho.py --sources wikiann --variant train-rho0.5-1
step "done"
