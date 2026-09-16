#!/usr/bin/env bash
# Wait for the "aligngrid" grid to finish, then run the alignment-vs-F1 analysis.
#
#   tmux new-session -d -s alignchain \
#     "bash scripts/chain_alignment_analysis.sh 2>&1 | tee outputs/alignment_chain.log"
#
# This lives in the repo and runs from a detached tmux session, so it survives a
# VS Code drop or an agent restart. The previous version of this waiter ran as a
# child of a Claude Code session and executed a copy of the analysis out of that
# session's scratchpad -- both die with the session, and the scratchpad copy can
# go stale against analysis/scripts/.
set -u
cd "$(dirname "$0")/.."

echo "[$(date -Is)] waiting for tmux session 'aligngrid'"
while tmux has-session -t aligngrid 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] grid finished; running alignment_vs_f1"

.venv/bin/python -W ignore analysis/scripts/alignment_vs_f1.py 2>&1 | tee outputs/alignment_vs_f1.log
echo "[$(date -Is)] ALIGNMENT_CHAIN_EXIT=${PIPESTATUS[0]}"
