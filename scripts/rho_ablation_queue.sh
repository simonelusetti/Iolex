#!/usr/bin/env bash
# rho ablation: the three home-task selectors retrained with
# model.selector.condition_on_rho=false (label suffix /no-rho), otherwise the
# identical experiment configs; then the home-task evaluations, seed overlap,
# and the comparison figure. Starts after the keep_last CoNLL training.
#
# Resumable like token_queue.sh: each stage verifies its result before writing a
# done-marker in outputs/rho_ablation/; reruns skip done stages and training
# resumes from checkpoints. "STAGE_FAILED <stage>" stops it; "QUEUE_DONE" ends it.
#
#   tmux new-session -d -s norho "scripts/rho_ablation_queue.sh 2>&1 | tee -a outputs/rho_ablation.log"
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MARK=outputs/rho_ablation; mkdir -p $MARK
F=".venv/bin/forge --config-dir old/conf --config-name config -M train grid"
COMMON="runtime.grid=true train.continue=true model.selector.condition_on_rho=false runtime.device=cuda"
PY=".venv/bin/python -W ignore"
PYTHIA_MIB=6000
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
step () { echo "[$(date -Is)] $*"; }
runs () {
    $PY - "$1" "$2" <<'PY'
import sys; sys.path.insert(0, ".")
from utils.forge_paths import discover_strategies
ds, label = sys.argv[1], sys.argv[2]
n = [len(s["runs"]) for s in discover_strategies(ds) if s["label"] == label and s["task"] == "rationale"]
print(n[0] if n else 0)
PY
}
prune () {  # older-epoch checkpoints of finished runs (each keeps its latest); disk filled once
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
}
stage () {  # stage NAME CHECK A B -- CMD...   (CHECK: "runs DS LABEL" or "file PATH")
    local name=$1 kind=$2 a=$3 b=${4:-}; shift 4; [ "$1" = "--" ] && shift
    if [ -e $MARK/$name.done ]; then step "skip $name (done)"; return 0; fi
    step "start $name"
    "$@"
    local ok=0
    if [ "$kind" = runs ]; then n=$(runs "$a" "$b"); step "$name: $n/3 completed runs"; [ "$n" -ge 3 ] && ok=1
    else [ -s "$a" ] && ok=1; fi
    if [ $ok = 1 ]; then touch $MARK/$name.done; step "done $name"; [ "$kind" = runs ] && prune
    else step "STAGE_FAILED $name"; exit 1; fi
}

until grep -q KEEP_LAST_DONE outputs/keeplast.log 2>/dev/null; do sleep 120; done

stage train_sbert runs stsb sbert/mean/no-rho -- \
    $F $COMMON data.dataset=stsb data.encoder.family=sbert data.encoder.pooling=mean \
    runtime.data.batch_size=16 --sweep runtime.seed=0,1,2

stage train_e5 runs beir_queries e5/mean/no-rho -- \
    $F $COMMON runtime.eval.stsb=false data.dataset=beir_queries data.encoder.family=e5 \
    data.encoder.pooling=mean model.keep_special=true +model.selector.keep_nonword=true \
    runtime.data.batch_size=16 --sweep runtime.seed=0,1,2

if [ ! -e $MARK/train_pythia.done ]; then
    until [ "$(free_mib)" -ge $PYTHIA_MIB ]; do
        step "waiting for $PYTHIA_MIB MiB of GPU memory for pythia ($(free_mib) free)"; sleep 300
    done
fi
stage train_pythia runs pile_causal llm/last/keep-last/next-token/no-rho -- \
    $F $COMMON runtime.eval.stsb=false data.dataset=pile_causal data.encoder.family=llm \
    data.encoder.pooling=last data.max_length=128 data.shuffle=true model.keep_special=true \
    +model.selector.keep_last=true +model.loss.objective=next_token_kl \
    model.optim.lr=3e-4 train.epochs=15 +train.gradient_clip=1.0 +train.selector_only_checkpoint=true \
    runtime.data.batch_size=8 --sweep runtime.seed=0,1,2

stage eval_sbert file outputs/home_tasks/sbert_stsb_norho.json "" -- \
    $PY analysis/scripts/home_task_sbert.py --no-rho
stage eval_e5 file outputs/home_tasks/e5_fiqa_norho.json "" -- \
    $PY analysis/scripts/home_task_e5.py --no-rho --sources in-domain
stage eval_pythia file outputs/home_tasks/pythia_pile_norho.json "" -- \
    $PY analysis/scripts/home_task_pythia_pile.py --no-rho
stage stability file outputs/home_tasks/stability_norho.json "" -- \
    $PY analysis/scripts/home_task_stability.py --no-rho
stage plot file analysis/reviews/rho_ablation.png "" -- \
    $PY analysis/scripts/rho_ablation_plot.py
step "QUEUE_DONE"
