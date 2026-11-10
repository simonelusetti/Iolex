#!/usr/bin/env bash
# Token-level (subword-piece) selectors on each encoder's home task, then their
# home-task evaluations -- the word-level experiments' exact configs plus
# +model.selector.unit=token (label suffix /tokens).
#
# Resumable: every stage verifies its own result before writing a done-marker
# in outputs/token_queue/, and a rerun skips finished stages. Training resumes
# from the last checkpoint (train.continue=true). A stage that fails its check
# prints "STAGE_FAILED <stage>" and the queue stops there, so it can be fixed
# and rerun; the last line of a clean run is "QUEUE_DONE".
#
#   tmux new-session -d -s tokens "scripts/token_queue.sh 2>&1 | tee -a outputs/tokens.log"
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MARK=outputs/token_queue; mkdir -p $MARK
F=".venv/bin/forge --config-dir old/conf --config-name config -M train grid"
COMMON="runtime.grid=true train.continue=true +model.selector.unit=token runtime.device=cuda"
PY=".venv/bin/python -W ignore"
PYTHIA_MIB=6000        # measured peak 4.6 GiB + CUDA context
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
step () { echo "[$(date -Is)] $*"; }
runs () {   # completed runs of a label (forge's grid swallows failures, so count)
    $PY - "$1" "$2" <<'PY'
import sys; sys.path.insert(0, ".")
from utils.forge_paths import discover_strategies
ds, label = sys.argv[1], sys.argv[2]
n = [len(s["runs"]) for s in discover_strategies(ds) if s["label"] == label and s["task"] == "rationale"]
print(n[0] if n else 0)
PY
}
stage () {  # stage NAME CHECK -- CMD...   (CHECK: "runs DS LABEL" or "file PATH")
    local name=$1 kind=$2 a=$3 b=${4:-}; shift 4; [ "$1" = "--" ] && shift
    if [ -e $MARK/$name.done ]; then step "skip $name (done)"; return 0; fi
    step "start $name"
    "$@"
    local ok=0
    if [ "$kind" = runs ]; then n=$(runs "$a" "$b"); step "$name: $n/3 completed runs"; [ "$n" -ge 3 ] && ok=1
    else [ -s "$a" ] && ok=1; fi
    if [ $ok = 1 ]; then touch $MARK/$name.done; step "done $name"
    else step "STAGE_FAILED $name"; exit 1; fi
}

stage train_sbert runs stsb sbert/mean/tokens -- \
    $F $COMMON data.dataset=stsb data.encoder.family=sbert data.encoder.pooling=mean \
    runtime.data.batch_size=16 --sweep runtime.seed=0,1,2

stage train_e5 runs beir_queries e5/mean/tokens -- \
    $F $COMMON runtime.eval.stsb=false data.dataset=beir_queries data.encoder.family=e5 \
    data.encoder.pooling=mean model.keep_special=true +model.selector.keep_nonword=true \
    runtime.data.batch_size=16 --sweep runtime.seed=0,1,2

if [ ! -e $MARK/train_pythia.done ]; then
    until [ "$(free_mib)" -ge $PYTHIA_MIB ]; do
        step "waiting for $PYTHIA_MIB MiB of GPU memory for pythia ($(free_mib) free)"; sleep 300
    done
fi
stage train_pythia runs pile_causal llm/last/keep-last/next-token/tokens -- \
    $F $COMMON runtime.eval.stsb=false data.dataset=pile_causal data.encoder.family=llm \
    data.encoder.pooling=last data.max_length=128 data.shuffle=true model.keep_special=true \
    +model.selector.keep_last=true +model.loss.objective=next_token_kl \
    model.optim.lr=3e-4 train.epochs=15 +train.gradient_clip=1.0 +train.selector_only_checkpoint=true \
    runtime.data.batch_size=8 --sweep runtime.seed=0,1,2

stage eval_sbert file outputs/home_tasks/sbert_stsb_tokens.json "" -- \
    $PY analysis/scripts/home_task_sbert.py --unit token
stage eval_sbert_curve file outputs/stsb_word/stsb__sbert_mean_tokens.json "" -- \
    $PY analysis/scripts/stsb_word_curves.py --dataset stsb --label sbert/mean/tokens
stage eval_e5 file outputs/home_tasks/e5_fiqa_tokens.json "" -- \
    $PY analysis/scripts/home_task_e5.py --unit token --sources in-domain
stage eval_pythia file outputs/home_tasks/pythia_pile_tokens.json "" -- \
    $PY analysis/scripts/home_task_pythia_pile.py --unit token
step "QUEUE_DONE"
