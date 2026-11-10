#!/usr/bin/env bash
# Token-level (subword-piece) selectors on each encoder's home task, to compare
# with the word-level ones: the identical experiment config of each existing
# home-task selector plus +model.selector.unit=token (label suffix /tokens).
#   sbert/mean/tokens                     STS-B           (as 0d6378d1)
#   e5/mean/tokens                        BEIR queries    (as a937f599, E5 as designed)
#   llm/last/keep-last/next-token/tokens  Pile causal     (as 4980a41d)
# then the home-task evaluations with --unit token (rho counts pieces for the
# selector and the random baseline alike), and the 20-point STS-B curve.
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=".venv/bin/forge --config-dir old/conf --config-name config -M train grid"
COMMON="runtime.grid=true train.continue=true +model.selector.unit=token runtime.device=cuda"
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
count () {   # completed runs of a label -- forge's grid swallows failures
    .venv/bin/python -W ignore - "$1" "$2" <<'PY'
import sys; sys.path.insert(0, ".")
from utils.forge_paths import discover_strategies
ds, label = sys.argv[1], sys.argv[2]
n = [len(s["runs"]) for s in discover_strategies(ds) if s["label"] == label and s["task"] == "rationale"]
print(f"  {ds} {label}: {n[0] if n else 0} completed runs")
PY
}
step () { echo "[$(date -Is)] $*"; }

step "train sbert/mean/tokens on stsb"
$F $COMMON data.dataset=stsb data.encoder.family=sbert data.encoder.pooling=mean \
    runtime.data.batch_size=16 --sweep runtime.seed=0,1,2
count stsb sbert/mean/tokens

step "train e5/mean/tokens on beir_queries"
$F $COMMON runtime.eval.stsb=false data.dataset=beir_queries data.encoder.family=e5 \
    data.encoder.pooling=mean model.keep_special=true +model.selector.keep_nonword=true \
    runtime.data.batch_size=16 --sweep runtime.seed=0,1,2
count beir_queries e5/mean/tokens

until [ "$(free_mib)" -ge 8000 ]; do
    step "waiting for 8000 MiB of GPU memory for pythia ($(free_mib) free)"; sleep 300
done
step "train llm/last/keep-last/next-token/tokens on pile_causal ($(free_mib) MiB free)"
$F $COMMON runtime.eval.stsb=false data.dataset=pile_causal data.encoder.family=llm \
    data.encoder.pooling=last data.max_length=128 data.shuffle=true model.keep_special=true \
    +model.selector.keep_last=true +model.loss.objective=next_token_kl \
    model.optim.lr=3e-4 train.epochs=15 +train.gradient_clip=1.0 +train.selector_only_checkpoint=true \
    runtime.data.batch_size=8 --sweep runtime.seed=0,1,2
count pile_causal llm/last/keep-last/next-token/tokens

step "evaluate: home tasks, token level"
.venv/bin/python -W ignore analysis/scripts/home_task_sbert.py --unit token
.venv/bin/python -W ignore analysis/scripts/stsb_word_curves.py --dataset stsb --label sbert/mean/tokens
.venv/bin/python -W ignore analysis/scripts/home_task_e5.py --unit token --sources in-domain
.venv/bin/python -W ignore analysis/scripts/home_task_pythia_pile.py --unit token
step "done"
