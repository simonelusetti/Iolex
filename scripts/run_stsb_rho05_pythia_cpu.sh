#!/usr/bin/env bash
# Pythia part of run_stsb_rho05.sh, on CPU: the GPU is held by another user's
# long-running jobs (20.5 GB). STS-B sentences are short (p90 = 13 tokens), so a
# step is ~0.5 s on 18 threads -- roughly 2-3 h per seed. runtime.* is outside
# the signature, so this joins the same experiment a GPU run would have.
set -u
cd /home/simonelusetti/Iolex
echo "[$(date -Is)] --- pythia (llm/last), CPU, 18 threads ---"
.venv/bin/forge --config-dir old/conf --config-name config -M train grid \
    data.dataset=stsb data.encoder.family=llm data.encoder.pooling=last \
    'model.loss.sweep_range=[0.5,1.0,6]' \
    runtime.device=cpu runtime.threads=18 runtime.grid=true train.continue=true \
    runtime.data.batch_size=8 --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT_llm=$?"
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
echo "[$(date -Is)] done"
