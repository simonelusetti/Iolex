#!/usr/bin/env bash
# NER probe with the selector's persistence a appended to each word's input,
# then the baseline with a permuted within each sentence. conll2003; sbert, e5, pythia.
#
#   tmux new-session -d -s salprobe \
#     "bash scripts/run_saliency_probe.sh 2>&1 | tee outputs/saliency_probe.log"
#
# Same MLP, optimiser, epochs and batch size (32) as the plain probes; the only
# change is input dim D -> D+1. a is computed on the fly from the family's
# trained selectors (5 seeds x rho 0.1..0.9, hard top-k), so it exists for the
# train split too (tagger/saliency.py). Checked against the selection logs on the
# test split: identical at the selectors' own batch size (16); at 32 padding
# flips near-ties on 0.3% (sbert) to 2.5% (e5) of words.
#
# Stored as outputs/probe/conll2003/<family>+a and <family>+a_perm; the plain
# probes are untouched. e5 uses the retrained, prefix-kept selectors.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for MODE in a perm; do
    echo "[$(date -Is)] === saliency=$MODE ==="
    for FAM in sbert e5 llm; do
        echo "[$(date -Is)] --- conll2003 / $FAM / saliency=$MODE ---"
        .venv/bin/python -m tagger conll2003 --family "$FAM" --seeds 0,1,2 --device cuda \
            --saliency "$MODE" --set runtime.data.batch_size=32
        echo "[$(date -Is)] EXIT_${FAM}_${MODE}=$?"
    done
done

echo "[$(date -Is)] === macro F1, plain vs +a vs +a_perm ==="
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
import numpy as np, tagger
from tagger.tagging import headline_f1
for fam in ("sbert", "e5", "llm"):
    row = []
    for key in (fam, f"{fam}+a", f"{fam}+a_perm"):
        f = [headline_f1(r) for r in tagger.load("conll2003", key)]
        row.append(f"{key}: {np.mean(f):.4f} (n={len(f)})" if f else f"{key}: -")
    print("  " + "   ".join(row))
PY
echo "[$(date -Is)] done"
