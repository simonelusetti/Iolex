#!/usr/bin/env bash
# Waits for the running STS-B evaluation to finish, then sweeps the trapezoid's
# lower end: 0.1-1.0, 0.2-1.0, ... 0.9-1.0, all from the same cosines.
set -u
cd /home/simonelusetti/Iolex
# NB: do not pgrep -f for the command string -- the tmux SERVER inherits the
# command line of whichever new-session started it, so the pattern matches
# forever and the loop never exits. Wait on the session instead.
while tmux has-session -t "${WAIT_FOR:-conllsel}" 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] previous run done, starting span-start sweep"
.venv/bin/python -W ignore analysis/scripts/stsb_both_sides.py \
    --figure spans --families sbert,e5,llm --random-runs 3 \
    --rhos 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    --trap-spans 0.1-1.0,0.2-1.0,0.3-1.0,0.4-1.0,0.5-1.0,0.6-1.0,0.7-1.0,0.8-1.0,0.9-1.0
echo "[$(date -Is)] span-start sweep done"
