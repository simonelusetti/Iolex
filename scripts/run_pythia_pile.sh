#!/usr/bin/env bash
# Three stabilized Pythia selectors on Pythia's own causal-LM task and corpus,
# followed by a matched held-out Pile test evaluation.
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
if [ "$(free_mib)" -ge 8000 ]; then
    DEV="runtime.device=cuda"
else
    DEV="runtime.device=cpu runtime.threads=18"
fi

echo "[$(date -Is)] Pythia/Pile causal selector: 3 seeds, $DEV ($(free_mib) MiB free)"
.venv/bin/forge --config-dir old/conf --config-name config -M train grid \
    runtime.grid=true train.continue=true runtime.eval.stsb=false \
    data.dataset=pile_causal data.encoder.family=llm data.encoder.pooling=last \
    data.max_length=128 data.shuffle=true model.keep_special=true \
    +model.selector.keep_last=true +model.loss.objective=next_token_kl \
    model.optim.lr=3e-4 train.epochs=15 \
    +train.gradient_clip=1.0 +train.selector_only_checkpoint=true \
    $DEV runtime.data.batch_size=8 --sweep runtime.seed=0,1,2
train_status=$?
echo "[$(date -Is)] training exit=$train_status"
if [ "$train_status" -ne 0 ]; then
    exit "$train_status"
fi

echo "[$(date -Is)] matched official Pile test evaluation"
eval_device="${DEV#runtime.device=}"
eval_device="${eval_device%% *}"
.venv/bin/python -W ignore analysis/scripts/home_task_pythia_pile.py --device "$eval_device" --bs 8
eval_status=$?
echo "[$(date -Is)] evaluation exit=$eval_status"
exit "$eval_status"
