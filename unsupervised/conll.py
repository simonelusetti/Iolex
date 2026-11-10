"""CoNLL-2003 English for the unsupervised NER experiments.

Words and gold IOB2 tags per split (train / validation / test, kept separate --
the project's own loader merges validation into test), entity-level scoring,
and the cluster -> type matching unsupervised NER is evaluated with.

Unsupervised systems never see a gold tag. Their entity types are anonymous
clusters; following Luo et al. (2020), and unsupervised POS induction before
them, each cluster is mapped to a gold type only for evaluation. Here that
mapping is one-to-one (Hungarian), fitted on predicted spans whose boundaries
exactly match a gold span.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

TYPES = ["PER", "ORG", "LOC", "MISC"]


def load(split: str) -> tuple[list[list[str]], list[list[str]]]:
    """(sentences, IOB2 tags) for one split, -DOCSTART- lines dropped."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("eriktks/conll2003", f"conll2003/{split}/0000.parquet",
                        repo_type="dataset", revision="refs/convert/parquet")
    d = load_dataset("parquet", data_files={split: f})[split]
    names = d.features["ner_tags"].feature.names
    sents, tags = [], []
    for words, t in zip(d["tokens"], d["ner_tags"]):
        if not words or words[0] == "-DOCSTART-":
            continue
        sents.append(list(words))
        tags.append([names[i] for i in t])
    return sents, tags


def spans(tags: list[str]) -> list[tuple[int, int, str]]:
    """IOB2 -> [(start, end_exclusive, type)]. An I- that cannot continue the
    current span starts a new one (the usual repair)."""
    out, start, typ = [], None, None
    for i, t in enumerate(tags + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):
            if start is not None:
                out.append((start, i, typ))
            start, typ = (None, None) if t == "O" else (i, t[2:])
    return out


def to_tags(n: int, sp: list[tuple[int, int, str]]) -> list[str]:
    tags = ["O"] * n
    for s, e, t in sp:
        tags[s] = f"B-{t}"
        for i in range(s + 1, e):
            tags[i] = f"I-{t}"
    return tags


def prf(n_pred: int, n_gold: int, n_ok: int) -> dict:
    p = n_ok / n_pred if n_pred else 0.0
    r = n_ok / n_gold if n_gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "n_pred": n_pred, "n_gold": n_gold}


def span_scores(gold_tags, pred_spans) -> dict:
    """Unlabeled span detection: exact boundaries, type ignored."""
    g = {(i, s, e) for i, t in enumerate(gold_tags) for s, e, _ in spans(t)}
    p = {(i, s, e) for i, sp in enumerate(pred_spans) for s, e, *_ in sp}
    return prf(len(p), len(g), len(p & g))


def typed_scores(gold_tags, pred_spans) -> dict:
    """Entity-level micro P/R/F1, exact boundaries and type (conlleval
    semantics), plus per-type F1. pred_spans carry gold type names."""
    g = {(i, s, e, t) for i, tg in enumerate(gold_tags) for s, e, t in spans(tg)}
    p = {(i, s, e, t) for i, sp in enumerate(pred_spans) for s, e, t in sp}
    out = prf(len(p), len(g), len(p & g))
    out["per_type"] = {T: prf(sum(x[3] == T for x in p), sum(x[3] == T for x in g),
                              sum(x[3] == T for x in p & g))["f1"] for T in TYPES}
    return out


def match_clusters(gold_tags, pred_spans) -> dict[int, str]:
    """One-to-one cluster -> type map maximising exact-boundary agreement.

    pred_spans: per sentence [(start, end, cluster_id)]. Uses gold labels --
    the evaluation-only step every unsupervised NER paper performs."""
    gold = {(i, s, e): t for i, tg in enumerate(gold_tags) for s, e, t in spans(tg)}
    clusters = sorted({c for sp in pred_spans for *_, c in sp})
    C = np.zeros((len(clusters), len(TYPES)))
    for i, sp in enumerate(pred_spans):
        for s, e, c in sp:
            t = gold.get((i, s, e))
            if t is not None:
                C[clusters.index(c), TYPES.index(t)] += 1
    rows, cols = linear_sum_assignment(-C)
    mapping = {clusters[r]: TYPES[c] for r, c in zip(rows, cols)}
    # more clusters than types: the rest take their majority type
    for k, c in enumerate(clusters):
        if c not in mapping:
            mapping[c] = TYPES[int(C[k].argmax())]
    return mapping


def apply_mapping(pred_spans, mapping) -> list[list[tuple[int, int, str]]]:
    return [[(s, e, mapping[c]) for s, e, c in sp] for sp in pred_spans]


def gold_stats(tags) -> Counter:
    return Counter(t for tg in tags for *_, t in spans(tg))
