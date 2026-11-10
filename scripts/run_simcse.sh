#!/usr/bin/env bash
# SimCSE (pooling=cls) on conll2003, wikiann, conll2000, movie_rationales.
#
#   tmux new-session -d -s simcse \
#     "bash scripts/run_simcse.sh 2>&1 | tee outputs/simcse.log"
#
# Why SimCSE. The study had only two sentence encoders and neither is a clean
# instance of the thing being tested:
#   sbert = all-MiniLM-L6-v2, a 6-layer 384-dim distilled MiniLM, so every
#           sbert comparison varies depth and width as well as the objective;
#   e5    = a RETRIEVAL model used off-label, with a "query: " prefix that is the
#           sole cause of the "special"-token join bug.
# SimCSE (princeton-nlp/sup-simcse-bert-base-uncased) is purpose-built for
# sentence representation by supervised contrastive training on NLI, has no
# prefix convention, and is a full BERT-base stack on the same WordPiece vocab --
# verified to produce token ids identical to bert, so it shares the bert/electra
# null distribution cell for cell.
#
# POOLING=CLS, which is what SimCSE was trained against. Two changes were needed:
#   * a `cls` strategy in masked_pool (position 0), which requires
#     model.keep_special=true, since strip_special_tokens() otherwise deletes the
#     [CLS] token and position 0 would be the first ordinary word;
#   * an exemption in RationaleSelectorModel so attended non-word positions keep
#     z = 1. scatter_word_to_tokens multiplies by is_word, so [CLS] and [SEP]
#     were coming back as 0 and would have been masked out of attention at every
#     rho -- CLS built from a pattern in which CLS itself was invisible.
#     Selection is defined over words and the budget counts words, so specials
#     are not the selector's to drop. A measured no-op for keep_special=false:
#     exactly 0 of 4175 attended positions on conll2003 are non-words there.
# The loss therefore compares CLS-of-the-full-sentence against
# CLS-of-the-suppressed-sentence, which is the comparison asked for.
#
# The PROBES keep the default keep_special=false, matching every other probe in
# the store -- the probe gathers word-level states and never sees [CLS], so this
# costs nothing and keeps them comparable. The consequence is that the selection
# log carries "special" entries the probe has no row for, so any join must drop
# them; selector_scalar() and trajectories() already do.
#
# Config is old/conf/config.yaml -- the root one was replaced by the NER tagger's
# and has no task/forge/selector blocks. Entry point train.py is at the root.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -Is)] === 1/2  selectors: simcse / mean, seeds 0,1,2 ==="
for DS in conll2003 wikiann conll2000 movie_rationales; do
    echo "[$(date -Is)] --- selector $DS ---"
    .venv/bin/forge --config-dir old/conf --config-name config -M train grid \
        data.dataset="$DS" data.encoder.family=simcse data.encoder.pooling=cls \
        model.keep_special=true \
        runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 \
        train.continue=true \
        --sweep runtime.seed=0,1,2
    echo "[$(date -Is)] SELECTOR_${DS}_EXIT=$?"
done

echo "[$(date -Is)] === 2/2  probes ==="
for DS in conll2003 wikiann conll2000; do
    echo "[$(date -Is)] --- tagger $DS / simcse ---"
    .venv/bin/python -m tagger "$DS" --family simcse --seeds 0,1,2 --device cuda \
        --set runtime.data.batch_size=32
    echo "[$(date -Is)] TAGGER_${DS}_EXIT=$?"
done
# movie_rationales follows the README and the other movie probes: class-weighted.
echo "[$(date -Is)] --- tagger movie_rationales / simcse (class-weighted) ---"
.venv/bin/python -m tagger movie_rationales --family simcse --seeds 0,1,2 --device cuda \
    --class-weighted --set runtime.data.batch_size=16
echo "[$(date -Is)] TAGGER_movie_EXIT=$?"

echo "[$(date -Is)] === what landed ==="
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
import tagger
from utils.plot_grounding import discover_series
for ds in ("conll2003", "wikiann", "conll2000", "movie_rationales"):
    sel = next((len(s["bias_runs"]) for s in discover_series(ds)
                if s["label"] == "simcse/cls"), 0)
    print(f"  {ds:18s} selector seeds={sel}  probe seeds={len(tagger.load(ds,'simcse'))}")
PY
echo "[$(date -Is)] done"
