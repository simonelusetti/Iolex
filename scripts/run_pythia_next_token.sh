#!/usr/bin/env bash
# Pythia LAMBADA selectors trained on the task's own quantity: KL between the
# next-token distributions with the full and the compressed context
# (model.loss.objective=next_token_kl), label-free. keep_last as before, since
# the prediction is read at the last word. 3 seeds, then the LAMBADA evaluation
# (last word protected for every compressor) with this family next to the
# cosine-trained keep_last selectors and the STS-B ones.
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
# a worst-case training step (longest batch) measured at 6.4 GB on the GPU
if [ "$(free_mib)" -ge 8000 ]; then DEV="runtime.device=cuda"; else DEV="runtime.device=cpu runtime.threads=18"; fi
echo "[$(date -Is)] --- llm/last keep_last next_token_kl on lambada, $DEV ($(free_mib) MiB free) ---"
.venv/bin/forge --config-dir old/conf --config-name config -M train grid \
    runtime.grid=true train.continue=true runtime.eval.stsb=false \
    data.dataset=lambada data.encoder.family=llm data.encoder.pooling=last \
    +model.selector.keep_last=true +model.loss.objective=next_token_kl \
    $DEV runtime.data.batch_size=8 --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT=$?"
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
import sys; sys.path.insert(0, ".")
from utils.forge_paths import discover_strategies
for s in discover_strategies("lambada"):
    print(f"  lambada: {s['label']} -> {len(s['runs'])} completed runs")
PY
echo "[$(date -Is)] --- LAMBADA evaluation ---"
.venv/bin/python -W ignore analysis/scripts/home_task_pythia.py --keep-last --sources next-token,in-domain,stsb \
    --out-suffix _nexttoken 2>&1 | grep --line-buffered -vE "it/s|Generating"
echo "[$(date -Is)] done"
