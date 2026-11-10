#!/usr/bin/env bash
# 2x2 saliency-augmentation grid (positions x replacement words) for bert, roberta,
# electra (all mean pooling) on conll2003, plus a no-augmentation baseline retrained
# through the same code path. 3 seeds each.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FAMS=bert,roberta,electra
.venv/bin/python -W ignore analysis/scripts/saliency_augment.py \
    --families "$FAMS" --variants none --word-pools all --seeds 0,1,2
.venv/bin/python -W ignore analysis/scripts/saliency_augment.py \
    --families "$FAMS" --variants low,high --word-pools low,high --seeds 0,1,2
echo "[$(date -Is)] EXIT=$?"
