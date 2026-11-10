#!/usr/bin/env bash
# Build the results compendium once persistence_followup.py has finished.
set -u
cd "$(dirname "$0")/.."
echo "[$(date -Is)] waiting for persistence_followup"
while pgrep -f persistence_followup.py >/dev/null; do sleep 20; done
echo "[$(date -Is)] building compendium"
.venv/bin/python -W ignore analysis/scripts/compendium.py conll2003
echo "[$(date -Is)] EXIT=$?"
