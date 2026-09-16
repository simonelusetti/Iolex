#!/usr/bin/env bash
# Retrain every cached probe whose report survived but whose weights did not.
#
#   tmux new-session -d -s probes \
#     "bash scripts/run_retrain_probes.sh 2>&1 | tee outputs/retrain_probes.log"
#
# On 2026-09-13 at 11:50 a single bulk operation removed model.pth from 75
# probe seeds while leaving their report.json in place. The correlation tables
# only read reports, so they kept working; the word-level AUC re-scores every
# test word and needs the weights.
#
# The work list is derived from the store at launch, not hardcoded: a
# (dataset, family, seed) is retrained exactly when a report exists and
# tagger.tagging.checkpoint_for cannot resolve weights for it. Seeds that still
# have weights are never passed, because --retrain retrains every seed it is
# given. Probes that were never trained at all are not added.
#
# --retrain also rewrites report.json, so per-tag F1 behind the correlation
# tables will move slightly from the previous numbers. batch_size=32 matches
# the settings these probes were originally trained with.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PLAN=$(.venv/bin/python - <<'EOF'
import tagger
from collections import defaultdict
from tagger.tagging import STORE, checkpoint_for
missing = defaultdict(list)
for report_path in sorted(STORE.glob("*/*/seed*/report.json")):
    ds, fam = report_path.parts[-4], report_path.parts[-3]
    for report in tagger.load(ds, fam):
        if f"seed{report['seed']}" == report_path.parts[-2] and checkpoint_for(report) is None:
            missing[(ds, fam)].append(report["seed"])
for (ds, fam), seeds in sorted(missing.items()):
    print(ds, fam, ",".join(map(str, sorted(set(seeds)))))
EOF
)

echo "[$(date -Is)] plan (dataset family seeds):"
echo "$PLAN" | sed 's/^/    /'
echo "[$(date -Is)] $(echo "$PLAN" | grep -c .) pairs, $(echo "$PLAN" | awk '{n+=split($3,a,",")} END{print n}') seeds"

while read -r DS FAM SEEDS; do
    [ -z "$DS" ] && continue
    echo "[$(date -Is)] --- $DS / $FAM seeds $SEEDS ---"
    .venv/bin/python -m tagger "$DS" --family "$FAM" --seeds "$SEEDS" --device cuda --retrain \
        --set runtime.data.batch_size=32
    rc=$?
    echo "[$(date -Is)] PROBE_${DS}_${FAM}_EXIT=$rc"
done <<< "$PLAN"

# Exit codes above only cover the process; confirm the weights actually exist.
echo "[$(date -Is)] === still missing weights ==="
.venv/bin/python - <<'EOF'
import tagger
from tagger.tagging import STORE, checkpoint_for
left = [f"{p.parts[-4]}/{p.parts[-3]}/{p.parts[-2]}"
        for p in sorted(STORE.glob("*/*/seed*/report.json"))
        if any(f"seed{r['seed']}" == p.parts[-2] and checkpoint_for(r) is None
               for r in tagger.load(p.parts[-4], p.parts[-3]))]
print("none" if not left else "\n".join(left))
EOF
echo "[$(date -Is)] done"
