#!/usr/bin/env bash
# CoNLL-2000 and Movie Rationales: 3 selectors each, then the matching taggers.
#
#   tmux new-session -d -s c2kmr \
#     "bash scripts/run_conll2000_movie.sh 2>&1 | tee outputs/conll2000_movie.log"
#
# Both corpora had a bert probe but ZERO selector runs, so neither could enter any
# of the current analyses. The grid files pick the paper's three encoders
# (sbert, e5, llm), and only bert has a probe today, so the taggers below cover
# exactly the three the selectors use.
#
# llm runs at pooling=last, not the grid file's inherited default of mean: causal
# attention means only the final token has seen the whole sequence. This is also
# the first time llm is worth running here at all -- its selector could not
# receive gradient until gpt_token_embeddings was rewritten (see
# tests/test_gpt_mask.py), so the grid files' llm entries would previously have
# produced untrained selectors.
#
# movie_rationales taggers use --class-weighted, following the README. NOTE for
# any later prior-standardised analysis: class weighting puts that probe in Case 3
# of the scalar-performance protocol, where empirical class frequency is NOT the
# effective prior and pi_model cannot be read off the training counts.
#
# The config is old/conf/config.yaml -- the restructure replaced the root one with
# the NER tagger's, which has no task/forge/selector blocks.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sel () {   # dataset, families, pooling
    echo "[$(date -Is)] --- selector $1 / $2 / pooling=$3 ---"
    .venv/bin/forge --config-dir old/conf --config-name config -M train grid \
        data.dataset="$1" data.encoder.pooling="$3" \
        runtime.device=cuda runtime.grid=true runtime.data.batch_size=8 \
        train.continue=true \
        --sweep data.encoder.family="$2"
    echo "[$(date -Is)] SELECTOR_$1_$3_EXIT=$?"
}

echo "[$(date -Is)] === 1/2  selectors ==="
for DS in conll2000 movie_rationales; do
    sel "$DS" sbert,e5 mean
    sel "$DS" llm last
done

echo "[$(date -Is)] === 2/2  taggers ==="
for FAM in sbert e5 llm; do
    echo "[$(date -Is)] --- tagger conll2000 / $FAM ---"
    .venv/bin/python -m tagger conll2000 --family "$FAM" --seeds 0,1,2 --device cuda \
        --set runtime.data.batch_size=32
    echo "[$(date -Is)] TAGGER_conll2000_${FAM}_EXIT=$?"
done
for FAM in sbert e5 llm; do
    echo "[$(date -Is)] --- tagger movie_rationales / $FAM (class-weighted) ---"
    .venv/bin/python -m tagger movie_rationales --family "$FAM" --seeds 0,1,2 --device cuda \
        --class-weighted --set runtime.data.batch_size=16
    echo "[$(date -Is)] TAGGER_movie_${FAM}_EXIT=$?"
done

echo "[$(date -Is)] === what landed ==="
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
import tagger
from utils.plot_grounding import discover_series
for ds in ("conll2000", "movie_rationales"):
    print(f"{ds}:")
    for s in discover_series(ds):
        print(f"   selector {s['label']:18s} runs={len(s['bias_runs'])}")
    for fam in ("bert", "sbert", "e5", "llm"):
        n = len(tagger.load(ds, fam))
        if n:
            print(f"   probe    {fam:18s} seeds={n}")
PY
echo "[$(date -Is)] done"
