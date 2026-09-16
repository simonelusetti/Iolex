# Iolex NER experiments

This repository now contains the token-classification work: the reusable NER
tagger, frozen-BERT layer probes, word-level entropy analyses, and empirical
context-mask searches. The former selector/oracle/pooling project is preserved
under [`old/`](old/README.md).

## Tagger

Train or load one cached probe per dataset and encoder:

```bash
python -m tagger wikiann fewnerd --family bert --seeds 0 --device cuda
```

Train a separate probe on every BERT layer:

```bash
python -m tagger.layers fewnerd --device cuda
python -m utils.word_layer_analysis fewnerd --device cuda
```

## Empirical context masks

The current Few-NERD analysis uses focal-relative word positions. A competing
occurrence accumulates evidence

```text
sum(1 / abs(offset) for selected positions whose words differ)
```

and ceases to match at evidence 1. Masks minimize token count, then total
distance, then maximize same-label support; all complete ties are retained.

```bash
python -m utils.weighted_occurrence_masks \
  --device cpu --workers 16 \
  --output analysis/context_masks/fewnerd_weighted_occurrences.sqlite
```

The SQLite output is restartable and self-contained: it stores sentences,
token occurrences, labels, statuses, masks, and their matching occurrence IDs.

## Layout

- `tagger/`: probe model, training cache, and BERT-layer sweep.
- `src/data.py`, `src/datasets_builders.py`: labeled dataset loading.
- `src/sentence.py`: frozen token encoders used by the tagger.
- `utils/*occurrence_masks.py`: empirical context-mask searches.
- `utils/word_layer_analysis.py`: token/word performance by BERT layer.
- `analysis/`: generated NER results (ignored by git).
- `old/`: archived selector, oracle, pooling, STS-B, and paper-analysis code.

