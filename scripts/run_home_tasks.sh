#!/usr/bin/env bash
# Home-task sufficiency experiment, end to end and unattended:
#   1. selectors: E5 (as designed: [CLS]/[SEP] kept, "[CLS] query: ...") on BEIR
#      training queries and on STS-B; Pythia on LAMBADA validation contexts
#      (3 seeds each, default rho range). runtime.data.rebuild=true for E5 so
#      the tokenized cache carries the corrected prefix order.
#   2. evaluation: E5 retrieval (SciFact, NFCorpus, FiQA), Pythia LAMBADA.
#   3. figure + table.
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="runtime.grid=true train.continue=true runtime.eval.stsb=false"
E5="data.encoder.family=e5 data.encoder.pooling=mean +model.selector.keep_nonword=true model.keep_special=true runtime.data.rebuild=true runtime.data.batch_size=16"
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
F="--config-dir old/conf --config-name config -M train grid"

echo "[$(date -Is)] --- e5 (keep_special) on beir_queries ---"
.venv/bin/forge $F $COMMON data.dataset=beir_queries $E5 runtime.device=cuda --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT=$?"; prune
echo "[$(date -Is)] --- e5 (keep_special) on stsb ---"
.venv/bin/forge $F $COMMON data.dataset=stsb $E5 runtime.device=cuda --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT=$?"; prune
if [ "$(free_mib)" -ge 12000 ]; then DEV="runtime.device=cuda"; else DEV="runtime.device=cpu runtime.threads=18"; fi
echo "[$(date -Is)] --- llm on lambada, $DEV ($(free_mib) MiB free) ---"
.venv/bin/forge $F $COMMON data.dataset=lambada data.encoder.family=llm data.encoder.pooling=last \
    $DEV runtime.data.batch_size=8 --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT=$?"; prune

.venv/bin/python -W ignore - <<'PY'
import sys; sys.path.insert(0, ".")
from pathlib import Path
from omegaconf import OmegaConf
from utils.forge_paths import discover_strategies
for ds in ("beir_queries", "stsb", "lambada"):
    for s in discover_strategies(ds):
        if s["label"] in ("e5/mean", "llm/last"):
            ks = OmegaConf.select(OmegaConf.load(Path(s["runs"][0]).parent / "config.yaml"), "model.keep_special")
            print(f"  {ds}: {s['label']} keep_special={ks} -> {len(s['runs'])} completed runs")
PY

echo "[$(date -Is)] --- evaluation: E5 retrieval ---"
.venv/bin/python -W ignore analysis/scripts/home_task_e5.py 2>&1 | grep -vE "it/s|Generating|Downloading"
echo "[$(date -Is)] --- evaluation: Pythia LAMBADA ---"
.venv/bin/python -W ignore analysis/scripts/home_task_pythia.py 2>&1 | grep -vE "it/s|Generating|Downloading"
.venv/bin/python -W ignore analysis/scripts/home_task_plot.py > /dev/null && echo "[$(date -Is)] figure written"
echo "[$(date -Is)] done"
