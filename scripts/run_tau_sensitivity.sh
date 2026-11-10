#!/usr/bin/env bash
# Temperature sensitivity of the SBERT selector (STS-B home task): tau_rank and
# tau_gate one at a time, 4x below and above the defaults (0.05 / 0.2), 3 seeds,
# otherwise the default sbert/mean training. Then home-task evaluation and the figure.
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/python -W ignore"
for v in "tau_rank=0.0125" "tau_rank=0.2" "tau_gate=0.05" "tau_gate=0.8"; do
    echo "[$(date -Is)] --- training sbert/mean, $v ---"
    .venv/bin/forge --config-dir old/conf --config-name config -M train grid \
        data.dataset=stsb data.encoder.family=sbert data.encoder.pooling=mean \
        model.selector.$v runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 \
        runtime.eval.stsb=false train.continue=true --sweep runtime.seed=0,1,2
    echo "[$(date -Is)] training exit=$?"
    $PY - <<'PY'
import json, re
from pathlib import Path
for run in Path("outputs/xps").glob("*/*/"):
    md = run / "state/models"
    if md.is_dir() and (run / "meta.json").exists() and json.load(open(run / "meta.json")).get("status") == "done":
        cks = sorted(md.glob("model_*.pth"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
        for p in cks[:-1]:
            p.unlink()
PY
done
for v in "/tau-rank0.0125" "/tau-rank0.2" "/tau-gate0.05" "/tau-gate0.8"; do
    echo "[$(date -Is)] evaluating $v"
    $PY analysis/scripts/home_task_sbert.py --variant "$v" || { echo "[$(date -Is)] TAU_FAILED eval $v"; exit 1; }
done
$PY analysis/scripts/tau_sensitivity_plot.py || { echo "[$(date -Is)] TAU_FAILED plot"; exit 1; }
echo "[$(date -Is)] TAU_DONE"
