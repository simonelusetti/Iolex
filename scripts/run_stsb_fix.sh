#!/usr/bin/env bash
# Repairs for the STS-B selector set:
#   * e5 with the "query: " prefix kept (+model.selector.keep_nonword=true) -- the
#     first attempt trained without it and landed as "e5/mean/prefix-masked";
#   * pythia seed 2, whose first attempt died at epoch 9 when the disk filled.
# Prunes older-epoch checkpoints of finished runs after each family: only the last
# one is ever read, and pythia writes 1.4 GB per epoch.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

echo "[$(date -Is)] --- stsb / e5 / mean, prefix kept, seeds 0,1,2 ---"
.venv/bin/forge --config-dir old/conf --config-name config -M train grid \
    data.dataset=stsb data.encoder.family=e5 data.encoder.pooling=mean \
    +model.selector.keep_nonword=true \
    runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 \
    train.continue=true --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT_e5=$?"; prune

echo "[$(date -Is)] --- stsb / llm / last, seed 2 ---"
.venv/bin/forge --config-dir old/conf --config-name config -M train grid \
    data.dataset=stsb data.encoder.family=llm data.encoder.pooling=last \
    runtime.device=cuda runtime.grid=true runtime.data.batch_size=8 \
    train.continue=true --sweep runtime.seed=2
echo "[$(date -Is)] EXIT_llm=$?"; prune
echo "[$(date -Is)] done"
