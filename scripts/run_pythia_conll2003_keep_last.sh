#!/usr/bin/env bash
# Pythia (llm/last) CoNLL-2003 selectors with the last word protected
# (+model.selector.keep_last=true): the experiment behind the paper's CoNLL
# heatmap and NER-correlation scatter, with the corrected last-token handling.
# Otherwise identical to scripts/run_pythia_conll2003.sh's selector step.
set -u
cd /home/simonelusetti/Iolex
.venv/bin/forge --config-dir old/conf --config-name config -M train grid \
    data.dataset=conll2003 data.encoder.family=llm data.encoder.pooling=last \
    +model.selector.keep_last=true \
    runtime.device=cuda runtime.grid=true runtime.data.batch_size=8 \
    train.continue=true \
    --sweep runtime.seed=0,1,2
.venv/bin/python -W ignore - <<'PY'
import sys; sys.path.insert(0, ".")
from utils.forge_paths import discover_strategies
n = [len(s["runs"]) for s in discover_strategies("conll2003") if s["label"] == "llm/last/keep-last"]
print(f"llm/last/keep-last on conll2003: {n[0] if n else 0}/3 completed runs")
PY
echo "[$(date -Is)] KEEP_LAST_DONE"
