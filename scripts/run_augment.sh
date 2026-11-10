#!/usr/bin/env bash
# Selector-guided augmentation on STS-B -- pilot, then the full grid, unattended.
#   1. views for sbert; pilot: sbert, 25% of train, seeds 0-2, all conditions
#   2. wait for the conll2003 rho>=0.5 job (shared GPU)
#   3. views for e5 and pythia; full grid: sbert, e5, pythia x {25%, 100%} x seeds 0-4
#      x {full, random, sel_full, sel_half}, plus the untrained reference;
#      the summary is refreshed after every block.
# A run whose result exists is skipped, so re-running this script resumes.
# Each run first waits for enough free GPU memory; a failed run is logged as
# FAILED and the queue carries on.
set -u
cd /home/simonelusetti/Iolex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LC_ALL=C
PY=".venv/bin/python -W ignore"
CONDS="full random sel_full sel_half"
declare -A NEED=([sbert]=4000 [e5]=8000 [llm]=14000)
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu () {
    while [ "$(free_mib)" -lt "$1" ]; do
        echo "[$(date -Is)] waiting for GPU: $(free_mib) MiB free, need $1"; sleep 120
    done
}
run () {   # family condition fraction seed
    wait_gpu "${NEED[$1]}"
    $PY analysis/scripts/augment_finetune.py --family "$1" --condition "$2" --fraction "$3" --seed "$4" 2>&1 \
        | grep -E '^\{|^exists' | tail -1
    [ -f "outputs/augment/results/$1_$2_f$3_seed$4.json" ] || echo "[$(date -Is)] FAILED $1 $2 f$3 seed$4"
}
summary () { $PY analysis/scripts/augment_summary.py | tail -n 8; }
block () {  # family fraction seed...
    local fam=$1 frac=$2; shift 2
    echo "[$(date -Is)] --- $fam, fraction $frac, seeds $* ---"
    for seed in "$@"; do for c in $CONDS; do run "$fam" "$c" "$frac" "$seed"; done; done
    summary
}

echo "[$(date -Is)] === views for sbert ==="
$PY analysis/scripts/augment_views.py --families sbert 2>&1 | grep -E "wrote|exists"
run sbert none 0.25 0
block sbert 0.25 0 1 2
echo "[$(date -Is)] === PILOT DONE ==="

while tmux has-session -t conllrho05 2>/dev/null; do sleep 300; done
echo "[$(date -Is)] conll2003 job finished; starting the full grid"

wait_gpu 8000
echo "[$(date -Is)] === views for e5, pythia ==="
$PY analysis/scripts/augment_views.py --families e5,llm 2>&1 | grep -E "wrote|exists"
for fam in sbert e5 llm; do
    run "$fam" none 0.25 0
    block "$fam" 0.25 0 1 2 3 4
    block "$fam" 1 0 1 2 3 4
done
echo "[$(date -Is)] === ALL DONE ==="
