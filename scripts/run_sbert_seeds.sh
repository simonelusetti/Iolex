#!/usr/bin/env bash
# 10 more sbert/mean STS-B selector seeds (3..12), to see whether sbert's slight
# edge over the full sentence in the trapezoid figure is consistent or noise.
# Config is left at defaults so these join experiment 0d6378d1 (seeds 0..2).
# Then: prune older-epoch checkpoints, and redraw the span-start figure.
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -Is)] training sbert seeds 3-12"
.venv/bin/forge --config-dir old/conf --config-name config -M train grid \
    data.dataset=stsb data.encoder.family=sbert data.encoder.pooling=mean \
    runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 \
    train.continue=true --sweep runtime.seed=3,4,5,6,7,8,9,10,11,12
echo "[$(date -Is)] training exit=$?"

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

echo "[$(date -Is)] redrawing the span-start figure with all sbert seeds"
.venv/bin/python -W ignore analysis/scripts/stsb_both_sides.py \
    --figure spans --families sbert,e5,llm --random-runs 3 \
    --rhos 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    --trap-spans 0.1-1.0,0.2-1.0,0.3-1.0,0.4-1.0,0.5-1.0,0.6-1.0,0.7-1.0,0.8-1.0,0.9-1.0
echo "[$(date -Is)] done"
