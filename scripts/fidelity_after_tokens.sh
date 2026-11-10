#!/usr/bin/env bash
# Direct-fidelity evaluations that need the GPU or the token selectors: waits for
# the token queue (scripts/token_queue.sh) to finish, then E5 word level, then
# SBERT and E5 token level, then both figures. Last line of a clean run: FIDELITY_DONE.
set -u
cd /home/simonelusetti/Iolex
P=".venv/bin/python -W ignore analysis/scripts/home_task_fidelity.py"
until tail -1 outputs/tokens.log | grep -q QUEUE_DONE; do sleep 300; done
echo "[$(date -Is)] token queue done; fidelity starts"
$P compute --task e5 && \
$P compute --task sbert --unit token && \
$P compute --task e5 --unit token && \
$P plot && $P plot --unit token && echo "[$(date -Is)] FIDELITY_DONE" || echo "[$(date -Is)] FIDELITY_FAILED"
