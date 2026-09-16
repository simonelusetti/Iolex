"""Within-label, word-level AUC: does the selector keep the words the tagger gets right?

For every word of a dataset's test split:

    score    how strongly the selector keeps the word, read from the run's
             data/selection_log.npz. At rho="avg" it is the fraction of the
             logged rhos at which the word is kept; at a numeric rho it is the
             0/1 keep decision at that rho.
    outcome  whether the frozen tagger labels the word correctly. The tagger is
             the cached probe of the series' own token encoder, ensembled over
             its seeds (mean softmax, then argmax), so there is one well-defined
             right/wrong per word and no ties to break.

    AUC      P(score of a correctly tagged word > score of a mistagged one),
             ties counting one half. 0.5 = no relation; above 0.5 the selector
             keeps the words the tagger gets right.

The AUC is computed WITHIN each gold label and then averaged. A pooled AUC
would be dominated by the label mix -- O is both the easiest class and treated
differently by the selector -- and would restate the entity-vs-O split. The
within-label number asks a different question from the tag-level correlation:
not "are more-selected TAGS tagged better" but "inside one tag, are the words
the encoder keeps easier to tag". That is intended, and is what buys the
statistical power the tag-level test cannot have. Pooled and count-weighted
versions are reported alongside for reference only.

The join is the delicate part. The selection log stores words in eval-loader
order, and that order changed when the eval loader became length-sorted, so
older runs are in dataset order and newer ones are not. Which order a run used
is established, never assumed: the run's logged token ids must equal the
probe's token ids concatenated in exactly one of the two orders, and the gold
labels must then agree word for word. A run matching neither is skipped with
the reason, not joined approximately.

Per-word tagger correctness costs one encoder pass per (dataset, encoder), so
it is cached beside the probe as outputs/probe/<dataset>/<family>/word_correct.npz
and recomputed only when the probe checkpoints change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE_NAME = "word_correct.npz"
_PROBES: dict[tuple[str, str], tuple[dict | None, str]] = {}


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks, so ties count one half."""
    n1 = int(positive.sum())
    n0 = len(positive) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------------------
# Per-word tagger correctness
# ---------------------------------------------------------------------------

def _checkpoints(dataset: str, family: str) -> list[tuple[dict, Path]]:
    import tagger
    from tagger.tagging import checkpoint_for
    out = []
    for report in tagger.load(dataset, family):
        path = checkpoint_for(report)
        if path is not None:
            out.append((report, path))
    return out


def _load_probe(path: Path, device: str):
    import torch
    from tagger.model import MLPTagger
    state = torch.load(path, map_location=device)
    # Probes migrated out of the old NER runs are wrapped in a full run
    # checkpoint; freshly trained ones are a bare state_dict.
    if "model" in state and not any(k.startswith("net.") for k in state):
        state = state["model"]
    hidden, dim = state["net.0.weight"].shape
    model = MLPTagger(dim, num_tags=state["net.3.weight"].shape[0], hidden=hidden)
    model.load_state_dict(state)
    return model.to(device).eval()


def _fingerprint(checkpoints) -> str:
    return ";".join(f"seed{r['seed']}:{p.stat().st_mtime_ns}" for r, p in checkpoints)


def _compute(dataset: str, family: str, device: str, checkpoints) -> dict:
    import torch
    from omegaconf import OmegaConf
    from src.data import PAD_TAG, initialize_data
    from src.utils import configure_runtime, to_device
    from tagger.model import first_subword_mask, gather_word_level
    from tagger.tagging import tag_names

    cfg = OmegaConf.load(ROOT / "conf/config.yaml")
    cfg.data.dataset = dataset
    cfg.data.encoder.family = family
    cfg.data.encoder.pooling = "mean"      # pooling cannot reach a probe
    cfg.runtime.device = device
    cfg.runtime, _ = configure_runtime(cfg.runtime)
    dev = cfg.runtime.device
    # Exactly the data path the probe and the selector runs were built with.
    _, test_dl, encoder, *_ = initialize_data(
        cfg.data, cfg.runtime.data, None, device=dev,
        keep_special=bool(cfg.model.keep_special))

    tags = tag_names(dataset)
    to_idx = {str(i): i for i in range(len(tags))} | {t: i for i, t in enumerate(tags)}
    models = [_load_probe(p, dev) for _, p in checkpoints]
    order = list(test_dl.sampler)
    ids_rows, ok_rows, gold_rows = [None] * len(order), [None] * len(order), [None] * len(order)
    pg_rows = [None] * len(order)
    hits = np.zeros(len(models))
    total, offset = 0, 0

    with torch.no_grad():
        for batch in test_dl:
            b = to_device(dev, batch)
            label_ids = torch.tensor(
                [[-1 if v == PAD_TAG else to_idx[v] for v in seq] for seq in batch["labels"]],
                dtype=torch.long, device=dev)
            emb = encoder.token_embeddings(b["ids"], b["attn_mask"])
            word_emb, word_mask, word_labels = gather_word_level(emb, b["word_ids"], label_ids)
            probs = [torch.softmax(m(word_emb, word_mask), -1) for m in models]
            for k, p in enumerate(probs):
                hits[k] += ((p.argmax(-1) == word_labels) & word_mask).sum().item()
            total += int(word_mask.sum())
            ens = torch.stack(probs).mean(0)
            correct = ens.argmax(-1) == word_labels
            # Probability the ensemble puts on the TRUE label: a continuous
            # difficulty measure, where correct/incorrect is only its sign test.
            p_gold = ens.gather(-1, word_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            first = first_subword_mask(b["word_ids"])
            for row in range(b["ids"].shape[0]):
                n = int(word_mask[row].sum())
                target = order[offset + row]
                ids_rows[target] = b["ids"][row][first[row]].cpu().numpy().astype(np.int32)
                ok_rows[target] = correct[row, :n].cpu().numpy()
                pg_rows[target] = p_gold[row, :n].cpu().numpy().astype(np.float32)
                gold_rows[target] = word_labels[row, :n].cpu().numpy().astype(np.int16)
                assert len(ids_rows[target]) == n
            offset += b["ids"].shape[0]

    # Each probe seed must reproduce its own cached accuracy, or the per-word
    # outcomes below are not the probe that the F1 numbers describe.
    checks = []
    for (report, _), h in zip(checkpoints, hits):
        cached = float(report["token_level"]["accuracy"])
        checks.append(f"seed{report['seed']} {h / max(1, total):.4f} (cached {cached:.4f})")
    print(f"[auc] {dataset}/{family}: {total:,} words; probe accuracy " + ", ".join(checks),
          file=sys.stderr)

    return {
        "lengths": np.array([len(r) for r in ids_rows], dtype=np.int64),
        "token_id": np.concatenate(ids_rows),
        "correct": np.concatenate(ok_rows).astype(bool),
        "p_gold": np.concatenate(pg_rows).astype(np.float32),
        "gold": np.concatenate(gold_rows),
        "sorted_order": np.asarray(order, dtype=np.int64),
        "tags": np.asarray(tags),
    }


def probe_words(dataset: str, family: str, device: str | None) -> tuple[dict | None, str]:
    """Per-word correctness for (dataset, family), from cache when still valid."""
    key = (dataset, family)
    if key in _PROBES:
        return _PROBES[key]
    from tagger.tagging import store_dir

    import tagger
    checkpoints = _checkpoints(dataset, family)
    if not checkpoints:
        n_reports = len(tagger.load(dataset, family))
        why = (f"{n_reports} probe report(s) for {dataset}/{family} but no model.pth weights; "
               f"rebuild with `python3 -m tagger {dataset} --family {family} --retrain`"
               if n_reports else f"no probe for {dataset}/{family}")
        _PROBES[key] = (None, why)
        return _PROBES[key]
    fingerprint = _fingerprint(checkpoints)
    cache = store_dir(dataset, family, 0).parent / CACHE_NAME
    data = None
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["fingerprint"]) == fingerprint and "p_gold" in z.files:
            data = {k: z[k] for k in z.files}
    if data is None:
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        data = _compute(dataset, family, device, checkpoints)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, fingerprint=np.asarray(fingerprint), **data)

    starts = np.concatenate([[0], np.cumsum(data["lengths"])[:-1]])
    data["orders"] = {
        "dataset": np.arange(len(data["token_id"])),
        "length-sorted": np.concatenate(
            [np.arange(starts[r], starts[r] + data["lengths"][r]) for r in data["sorted_order"]]),
    }
    _PROBES[key] = (data, "")
    return _PROBES[key]


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_auc(run_path: Path, probe: dict, rho_labels: list[str], exclude: list[str],
            min_class: int) -> tuple[dict | None, str]:
    """{rho label: {macro, weighted, pooled, n_labels, ...}} for one selector run."""
    path = Path(run_path) / "data" / "selection_log.npz"
    if not path.exists():
        return None, "no selection_log.npz"
    z = np.load(path, allow_pickle=False)
    logged = z["token_id"].astype(np.int32)

    index, order = None, None
    for name, candidate in probe["orders"].items():
        if len(candidate) == len(logged) and np.array_equal(probe["token_id"][candidate], logged):
            index, order = candidate, name
            break
    if index is None:
        return None, "logged words match neither dataset nor length-sorted order"

    tags = [str(t) for t in probe["tags"]]
    to_idx = {str(i): i for i in range(len(tags))} | {t: i for i, t in enumerate(tags)}
    names = [str(n) for n in z["label_names"]]
    logged_gold = np.array([to_idx.get(names[i], -1) for i in z["label"]], dtype=np.int16)
    gold = probe["gold"][index]
    if not np.array_equal(logged_gold, gold):
        return None, f"gold labels disagree on {int((logged_gold != gold).sum())} words"

    correct = probe["correct"][index]
    selected = z["selected"].astype(np.float32)          # [R, N]
    logged_rhos = np.asarray(z["rho"], dtype=np.float64)
    keep = np.array([tags[g] not in exclude for g in range(len(tags))])[gold]

    out = {"order": order, "n_words": int(keep.sum()), "accuracy": float(correct[keep].mean())}
    for label in rho_labels:
        if label == "avg":
            score = selected.mean(axis=0)
        else:
            hit = np.flatnonzero(np.isclose(logged_rhos, float(label)))
            if not len(hit):
                continue
            score = selected[hit[0]]
        per_label = []
        for g in np.unique(gold[keep]):
            m = keep & (gold == g)
            n_ok = int(correct[m].sum())
            if n_ok < min_class or int(m.sum()) - n_ok < min_class:
                continue
            per_label.append((auc(score[m], correct[m]), int(m.sum())))
        values = np.array([v for v, _ in per_label])
        weights = np.array([n for _, n in per_label], dtype=float)
        out[label] = {
            "macro": float(values.mean()) if len(values) else float("nan"),
            "weighted": float((values * weights).sum() / weights.sum()) if len(values) else float("nan"),
            "pooled": auc(score[keep], correct[keep]),
            "n_labels": len(per_label),
            "n_labels_total": int(len(np.unique(gold[keep]))),
        }
    return out, ""
