#!/usr/bin/env bash
# Runs once the e5 keep_special=true selectors have finished training.
# Same span-start sweep as before, but E5 in its documented configuration
# ([CLS]/[SEP] kept, "query: " prefix); sbert and pythia are unchanged.
set -u
cd /home/simonelusetti/Iolex
while tmux has-session -t e5natural 2>/dev/null; do sleep 30; done
echo "[$(date -Is)] e5 training done, starting span-start sweep with E5 natural"
.venv/bin/python -W ignore analysis/scripts/stsb_both_sides.py \
    --figure spans --families sbert,e5,llm --random-runs 3 --keep-special-families e5 \
    --rhos 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    --trap-spans 0.1-1.0,0.2-1.0,0.3-1.0,0.4-1.0,0.5-1.0,0.6-1.0,0.7-1.0,0.8-1.0,0.9-1.0
echo "[$(date -Is)] done"
