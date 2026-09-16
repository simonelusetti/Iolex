#!/usr/bin/env bash
# Complete the (dataset x encoder) grid at pooling=mean, so the label-alignment
# design has enough paired comparisons to be worth running.
#
#   tmux new-session -d -s aligngrid \
#     "bash scripts/run_alignment_grid.sh 2>&1 | tee outputs/alignment_grid.log"
#
# The design: one point per (dataset, encoder), y = probe macro F1, x = how well
# selection separates the labels. Only WITHIN-dataset encoder pairs are valid
# evidence, so the number of pairs is what decides power. Today: 16 pairs, of
# which 3 have an F1 gap smaller than probe-seed noise. 13 of 16 concordant would
# be needed for p<0.05, and the observed 8-10 of 16 is indistinguishable from
# chance. Filling the grid to 5 encoders x 7 datasets gives 70 pairs, where a 65%
# tendency becomes detectable.
#
# Missing cells only; everything else is already in the store.
#   selectors  wikiann bert (deleted experiment), fewnerd sbert/e5,
#              fewnerd_fine electra/roberta/sbert/e5, fewnerd roberta seed 2
#   probes     fewnerd_fine electra/roberta/sbert/e5
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -Is)] === 1/3  probes: fewnerd_fine, the four non-bert encoders ==="
for FAM in electra roberta sbert e5; do
    echo "[$(date -Is)] --- fewnerd_fine / $FAM tagger ---"
    .venv/bin/python -m tagger fewnerd_fine --family "$FAM" --seeds 0,1,2 --device cuda \
        --set runtime.data.batch_size=32
    rc=$?
    echo "[$(date -Is)] TAGGER_fewnerd_fine_${FAM}_EXIT=$rc"
done

echo "[$(date -Is)] === 2/3  selectors, pooling=mean, seeds 0,1,2 ==="
run_sel () {   # dataset, comma-separated families, comma-separated seeds
    echo "[$(date -Is)] --- selectors $1 / $2 seeds $3 ---"
    .venv/bin/forge grid data.dataset="$1" data.encoder.pooling=mean \
        runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 \
        train.continue=true \
        --sweep data.encoder.family="$2" --sweep runtime.seed="$3"
    rc=$?
    echo "[$(date -Is)] SELECTOR_$1_EXIT=$rc"
}

# wikiann bert/mean: the experiment that went missing on 2026-09-13.
run_sel wikiann bert 0,1,2
run_sel fewnerd roberta 2
run_sel fewnerd sbert,e5 0,1,2
run_sel fewnerd_fine electra,roberta,sbert,e5 0,1,2

echo "[$(date -Is)] === 3/3  what the grid looks like now ==="
.venv/bin/python - <<'EOF'
import tagger
from utils.plot_grounding import discover_series
fams = ["bert", "electra", "roberta", "sbert", "e5"]
print(f"{'dataset':14s}" + "".join(f"{f:>10s}" for f in fams) + "   (selector seeds / probe seeds)")
total = 0
for ds in ["wikiann", "conll2003", "fewnerd", "fewnerd_fine", "ud_upos", "ud_deprel", "ud_discourse"]:
    got = {s["label"]: len(s["bias_runs"]) for s in discover_series(ds)}
    row, n_here = "", 0
    for f in fams:
        sel = got.get(f"{f}/mean", 0)
        probes = len(tagger.load(ds, f))
        row += f"{sel:>6d}/{probes:<4d}"
        n_here += 1 if (sel and probes) else 0
    total += n_here * (n_here - 1) // 2
    print(f"{ds:14s}{row}")
print(f"usable within-dataset encoder pairs: {total}")
EOF
echo "[$(date -Is)] done"
