#!/usr/bin/env bash
# STS-B selectors trained on budgets rho in {0.5, 0.6, ..., 1.0} only
# (model.loss.sweep_range=[0.5,1.0,6]; default is [0.1,1.0,10]), 3 seeds each
# for sbert, e5 and pythia. The range is part of the signature, so these are new
# experiments and resume can never pick up a default-range checkpoint.
#
# The GPU is shared: pythia needs ~25 GB, so it waits until that much is free
# instead of OOMing into someone else's job.
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RANGE='model.loss.sweep_range=[0.5,1.0,6]'
COMMON="runtime.device=cuda runtime.grid=true train.continue=true"

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
.venv/bin/forge --config-dir old/conf --config-name config -M train grid \
    data.dataset=stsb data.encoder.family=sbert data.encoder.pooling=mean "$RANGE" \
    $COMMON runtime.data.batch_size=16 --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT_sbert=$?"; prune

echo "[$(date -Is)] --- e5/mean, query: prefix kept ---"
.venv/bin/forge --config-dir old/conf --config-name config -M train grid \
    data.dataset=stsb data.encoder.family=e5 data.encoder.pooling=mean "$RANGE" \
    +model.selector.keep_nonword=true \
    $COMMON runtime.data.batch_size=16 --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT_e5=$?"; prune

NEED=26000   # MiB free before pythia starts
while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge "$NEED" ] && break
    echo "[$(date -Is)] pythia waiting: ${free} MiB free, need ${NEED}"
    sleep 300
done

echo "[$(date -Is)] --- pythia (llm/last) ---"
.venv/bin/forge --config-dir old/conf --config-name config -M train grid \
    data.dataset=stsb data.encoder.family=llm data.encoder.pooling=last "$RANGE" \
    $COMMON runtime.data.batch_size=8 --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT_llm=$?"; prune
echo "[$(date -Is)] done"
