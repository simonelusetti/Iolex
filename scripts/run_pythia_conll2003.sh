#!/usr/bin/env bash
# Pythia (llm) on CoNLL-2003 at pooling=last: the probe, then 3 seeds of the selector.
#
#   tmux new-session -d -s pythia \
#     "bash scripts/run_pythia_conll2003.sh 2>&1 | tee outputs/pythia_conll2003.log"
#
# This is the first llm run in the store that is not an oracle. It became
# possible by rewriting gpt_token_embeddings (src/sentence.py) as an explicit
# GPT-NeoX attention loop that adds log(attention_mask) to the scores, the way
# bert_token_embeddings already did. Previously the mask went to HuggingFace's
# own forward, which rebuilds it as a boolean additive bias and detaches it: the
# output came back with requires_grad=False, so the selector had no path to the
# loss and could not train. tests/test_gpt_mask.py pins both properties -- the
# rewrite is bitwise-identical to the HF forward on a 0/1 mask, and a fractional
# mask now receives gradient.
#
# pooling=last: causal attention means only the final token has seen the whole
# sequence, which is the standard way to embed a causal LM (GritLM / LLM2Vec /
# PromptEOL). It also makes the gradient fix load-bearing rather than incidental.
# masked_pool's `last` takes a hard index, so it "passes no gradient to the
# scores at all and the selector trains through the attention path alone" -- its
# own words. That attention path is exactly what was detached, so before the
# rewrite llm/last had NO gradient route whatsoever. Verified end to end: a soft
# z through gpt_token_embeddings + masked_pool(..., "last") yields a finite,
# non-zero gradient (|grad| mean 1.17).
#
# The cost is that these are not directly comparable with the sbert/mean and
# e5/mean points in the AUC plot, since pooling differs as well as the encoder.
#
# The config lives at old/conf/config.yaml: the restructure replaced the root
# conf/config.yaml with the NER tagger's, which has no `task`, `forge` or
# selector blocks. Pointing forge at the archived one leaves both projects
# working. The entry point train.py is at the root, where forge expects it.
#
# batch_size 8 because the mask forces SDPA onto its MATH kernel whenever
# gradients are tracked (see _sdpa_context), which materialises the full
# [B, heads, T, T] attention matrix across 24 layers.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -Is)] === 1/2  probe: conll2003 / llm, seeds 0,1,2 ==="
.venv/bin/python -m tagger conll2003 --family llm --seeds 0,1,2 --device cuda \
    --set runtime.data.batch_size=16
echo "[$(date -Is)] PROBE_EXIT=$?"

echo "[$(date -Is)] === 2/2  selector: conll2003 / llm / last, seeds 0,1,2 ==="
.venv/bin/forge --config-dir old/conf --config-name config -M train grid \
    data.dataset=conll2003 data.encoder.family=llm data.encoder.pooling=last \
    runtime.device=cuda runtime.grid=true runtime.data.batch_size=8 \
    train.continue=true \
    --sweep runtime.seed=0,1,2
echo "[$(date -Is)] SELECTOR_EXIT=$?"

echo "[$(date -Is)] === what landed ==="
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
import tagger
from utils.plot_grounding import discover_series
print("conll2003 series:")
for s in discover_series("conll2003"):
    print(f"   {s['label']:22s} runs={len(s['bias_runs'])}")
print("conll2003 llm probe seeds:", len(tagger.load("conll2003", "llm")))
PY
echo "[$(date -Is)] done"
