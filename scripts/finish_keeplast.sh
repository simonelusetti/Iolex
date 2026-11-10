#!/usr/bin/env bash
# Finish the Pythia CoNLL-2003 keep_last selectors (seed 2 resumes from epoch 9
# after its final checkpoint was truncated by a full disk), prune older-epoch
# checkpoints, then redraw the CoNLL heatmap and NER scatter with the corrected
# Pythia selector -- and with the old one (llm/last), for comparison.
set -u
cd /home/simonelusetti/Iolex
.venv/bin/forge --config-dir old/conf --config-name config -M train grid \
    data.dataset=conll2003 data.encoder.family=llm data.encoder.pooling=last \
    +model.selector.keep_last=true \
    runtime.device=cuda runtime.grid=true runtime.data.batch_size=8 \
    train.continue=true --sweep runtime.seed=2
.venv/bin/python -W ignore - <<'PY'
import json, re, sys
from pathlib import Path
sys.path.insert(0, ".")
from utils.forge_paths import discover_strategies
n = [len(s["runs"]) for s in discover_strategies("conll2003") if s["label"] == "llm/last/keep-last"]
print(f"llm/last/keep-last on conll2003: {n[0] if n else 0}/3 completed runs")
for run in Path("outputs/xps/f66bbfd1").glob("*/"):
    if (run / "meta.json").exists() and json.load(open(run / "meta.json")).get("status") == "done":
        cks = sorted((run / "state/models").glob("model_*.pth"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
        for p in cks[:-1]:
            p.unlink()
PY
P=".venv/bin/python -W ignore utils"
for v in keep_last old; do
    llm=$([ $v = keep_last ] && echo llm/last/keep-last || echo llm/last)
    $P/plot_bias_heatmaps.py --dataset conll2003 --ncols 1 --strategies sbert/mean,e5/mean,$llm \
        --output analysis/reviews/heat_conll2003_llm_$v.png
done
# scatter at rho=0.8, every encoder pinned to its seed-0 run
$P/plot_paper_figure_conll2003.py --sbert-bias-sig ea2273d2/b6d157a8 --e5-bias-sig c81a1fe9/45566591 \
    --llm-bias-sig f66bbfd1/8b8f7d38 --output analysis/reviews/paper_figure_conll2003_llm_keep_last.png
$P/plot_paper_figure_conll2003.py --sbert-bias-sig ea2273d2/b6d157a8 --e5-bias-sig c81a1fe9/45566591 \
    --llm-bias-sig 446f6018/01238274 --output analysis/reviews/paper_figure_conll2003_llm_old.png
echo "[$(date -Is)] FINISH_KEEPLAST_DONE"
