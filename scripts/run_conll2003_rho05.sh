#!/usr/bin/env bash
# conll2003 selectors trained on budgets rho in {0.5, ..., 1.0} only
# (model.loss.sweep_range=[0.5,1.0,6]), 3 seeds each for sbert, e5, pythia --
# otherwise identical to the existing conll2003 selectors (e5: query: prefix kept;
# pythia: last-token pooling). Then the span-start trapezoid figure on STS-B
# (a transfer test: the trapezoid needs gold similarity, which only STS-B has).
#
# The GPU is shared: each step checks free memory when it starts, and pythia
# falls back to CPU rather than OOM into someone else's job.
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RANGE='model.loss.sweep_range=[0.5,1.0,6]'
COMMON="data.dataset=conll2003 runtime.grid=true train.continue=true"
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

prune () {
    .venv/bin/python - <<'PY'
import json, re
from pathlib import Path
freed = 0
for run in Path("outputs/xps").glob("*/*/"):
    md = run / "state/models"
    if not md.is_dir() or not (run / "meta.json").exists():
        continue
    if json.load(open(run / "meta.json")).get("status") != "done":
        continue
    cks = sorted(md.glob("model_*.pth"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    for p in cks[:-1]:
        freed += p.stat().st_size; p.unlink()
print(f"pruned {freed/1024**3:.1f} GB of older-epoch checkpoints")
PY
}

echo "[$(date -Is)] --- sbert/mean ---"
.venv/bin/forge --config-dir old/conf --config-name config -M train grid $COMMON \
    data.encoder.family=sbert data.encoder.pooling=mean "$RANGE" \
    runtime.device=cuda runtime.data.batch_size=16 --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT_sbert=$?"; prune

echo "[$(date -Is)] --- e5/mean, query: prefix kept ---"
.venv/bin/forge --config-dir old/conf --config-name config -M train grid $COMMON \
    data.encoder.family=e5 data.encoder.pooling=mean "$RANGE" +model.selector.keep_nonword=true \
    runtime.device=cuda runtime.data.batch_size=16 --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT_e5=$?"; prune

if [ "$(free_mib)" -ge 26000 ]; then DEV="runtime.device=cuda"; else DEV="runtime.device=cpu runtime.threads=18"; fi
echo "[$(date -Is)] --- pythia (llm/last), $DEV ($(free_mib) MiB free) ---"
.venv/bin/forge --config-dir old/conf --config-name config -M train grid $COMMON \
    data.encoder.family=llm data.encoder.pooling=last "$RANGE" \
    $DEV runtime.data.batch_size=8 --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT_llm=$?"; prune

.venv/bin/python -W ignore - <<'PY'
import sys; sys.path.insert(0, ".")
from utils.forge_paths import discover_strategies
for s in discover_strategies("conll2003"):
    if s["label"].endswith("/train-rho0.5-1"):
        print(f"  {s['label']}: {len(s['runs'])} completed runs")
PY

if [ "$(free_mib)" -ge 12000 ]; then CPUF=""; else CPUF="--cpu-families llm"; fi
echo "[$(date -Is)] span-start figure on STS-B ($(free_mib) MiB free) ${CPUF}"
.venv/bin/python -W ignore analysis/scripts/stsb_both_sides.py \
    --figure spans --selector-dataset conll2003 --train-rho 0.5-1 --families sbert,e5,llm \
    $CPUF --threads 18 --random-runs 3 --rhos 0.5,0.6,0.7,0.8,0.9,1.0 \
    --trap-spans 0.5-1.0,0.6-1.0,0.7-1.0,0.8-1.0,0.9-1.0
echo "[$(date -Is)] two-row figure: trapezoid vs best single rho, conll2003 + wikiann"
.venv/bin/python -W ignore analysis/scripts/trapezoid_best_rho.py --sources conll2003,wikiann --variant train-rho0.5-1
echo "[$(date -Is)] done"
