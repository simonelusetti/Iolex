#!/usr/bin/env bash
# In-domain selectors for the home-task experiments (3 seeds each, default rho
# range): E5 on BEIR training-split queries, Pythia on LAMBADA validation contexts.
# Label-free, like every selector. The STS-B sweep at the end of training is
# skipped (runtime.eval.stsb=false) -- irrelevant here.
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="runtime.grid=true train.continue=true runtime.eval.stsb=false"
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

echo "[$(date -Is)] --- e5/mean on beir_queries ---"
.venv/bin/forge --config-dir old/conf --config-name config -M train grid $COMMON \
    data.dataset=beir_queries data.encoder.family=e5 data.encoder.pooling=mean \
    +model.selector.keep_nonword=true runtime.device=cuda runtime.data.batch_size=16 \
    --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT_e5=$?"; prune

if [ "$(free_mib)" -ge 12000 ]; then DEV="runtime.device=cuda"; else DEV="runtime.device=cpu runtime.threads=18"; fi
echo "[$(date -Is)] --- llm/last on lambada, $DEV ($(free_mib) MiB free) ---"
.venv/bin/forge --config-dir old/conf --config-name config -M train grid $COMMON \
    data.dataset=lambada data.encoder.family=llm data.encoder.pooling=last \
    $DEV runtime.data.batch_size=8 --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT_llm=$?"; prune

.venv/bin/python -W ignore - <<'PY'
import sys; sys.path.insert(0, ".")
from utils.forge_paths import discover_strategies
for ds in ("beir_queries", "lambada"):
    for s in discover_strategies(ds):
        print(f"  {ds}: {s['label']} -> {len(s['runs'])} completed runs")
PY
echo "[$(date -Is)] done"
