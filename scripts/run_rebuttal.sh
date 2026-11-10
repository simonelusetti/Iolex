#!/usr/bin/env bash
# Rebuttal: the two home-task reruns that need per-seed outputs (items 6 and 7).
set -u
cd /home/simonelusetti/Iolex
.venv/bin/python -W ignore analysis/scripts/rebuttal_efficiency.py
cp outputs/home_tasks/stability.json outputs/home_tasks/stability_before_rebuttal.json
.venv/bin/python -W ignore analysis/scripts/home_task_stability.py
echo "[$(date -Is)] REBUTTAL_DONE"
