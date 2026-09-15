"""Plot saved layer accuracies by training-label entropy and a word-only baseline.

    python -m utils.entropy_layer_curves wikiann fewnerd

No encoder inference or training. Reads word_layer_analysis tables and the
original training split, using the same truncation and word handling as probes.
"""
import argparse
from collections import defaultdict
import csv
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from scipy.stats import entropy, spearmanr

from src.data import (canonical_name, get_dataset, resolve_dataset,
                      shuffle_and_subset, strip_special_tokens)
from src.sentence import resolve_tokenizer
from tagger.tagging import ROOT
from utils.word_layer_analysis import write_csv

log = logging.getLogger(__name__)


def training_counts(report):
    cfg = OmegaConf.create(report["config"])
    tokenizer = resolve_tokenizer(cfg.data.encoder.family)
    raw = shuffle_and_subset(resolve_dataset(cfg.data.dataset), cfg.data.subset, cfg.data.shuffle)["train"]
    cached = shuffle_and_subset(get_dataset(cfg.data, cfg.runtime.data, tokenizer),
                                cfg.data.subset, cfg.data.shuffle)
    if not cfg.model.keep_special:
        cached = strip_special_tokens(cached, tokenizer)
    cached = cached["train"]
    assert len(raw) == len(cached)
    tags = report["tags"]
    lookup = {str(i): i for i in range(len(tags))} | {t: i for i, t in enumerate(tags)}
    counts = defaultdict(lambda: np.zeros(len(tags), dtype=int))
    for original, encoded in zip(raw, cached):
        ids = tokenizer(original["tokens"], is_split_into_words=True, truncation=True,
                        max_length=int(cfg.data.max_length))["input_ids"]
        if not cfg.model.keep_special:
            special = tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
            ids = [v for v, s in zip(ids, special) if not s]
        assert ids == encoded["ids"], "Raw/tokenized training sentences differ"
        seen = set()
        for wid, label in zip(encoded["word_ids"], encoded["labels"]):
            if wid is None or wid < 0 or wid in seen:
                continue
            seen.add(wid)
            tag = lookup[str(label)]
            assert tag == lookup[str(original["labels"][wid])]
            counts[original["tokens"][wid]][tag] += 1
    log.info("%s: counted %d training occurrences, %d word types", report["dataset"],
             sum(int(v.sum()) for v in counts.values()), len(counts))
    return counts


def analyze(dataset, seed, minimum):
    source = ROOT / "analysis/word_layers" / dataset / f"seed{seed}"
    report = json.loads((ROOT / "outputs/probe_layers" / dataset / "bert" /
                         f"seed{seed}/layer0/report.json").read_text())
    train_counts = training_counts(report)
    out = source / "contextual_gain"
    out.mkdir(parents=True, exist_ok=True)
    curves, correlations, table, coverage = [], [], [], []
    layers = np.arange(13)
    for regime in ("full", "binary"):
        with (source / f"word_types_{regime}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        words = [r["word"] for r in rows]
        train = np.array([train_counts.get(w, np.zeros(len(report["tags"]), dtype=int)) for w in words])
        global_train = np.stack(list(train_counts.values())).sum(axis=0)
        tags = report["tags"]
        if regime == "binary":
            outside = tags.index("O")
            train = np.column_stack((train[:, outside], train.sum(axis=1) - train[:, outside]))
            global_train = np.array([global_train[outside], global_train.sum() - global_train[outside]])
            tags = ["non_entity", "entity"]
        n_train = train.sum(axis=1)
        n_eval = np.array([int(r["occurrences"]) for r in rows])
        gold_counts = np.array([[int(r[f"label_count_{t}"]) for t in tags] for r in rows])
        accuracy = np.array([[float(r[f"accuracy_layer{i}"]) for i in layers] for r in rows])
        seen = n_train > 0
        h = np.full(len(words), np.nan)
        h[seen] = entropy(train[seen], axis=1, base=2)
        prediction = train.argmax(axis=1)
        prediction[~seen] = global_train.argmax()
        baseline = gold_counts[np.arange(len(words)), prediction] / n_eval
        gain = accuracy - baseline[:, None]
        middle = accuracy[:, 4:9].mean(axis=1)
        fields = ["word", "train_occurrences", "eval_occurrences", "train_entropy_bits",
                  "eval_entropy_bits", "baseline_label", "baseline_accuracy", "unseen_in_train"]
        fields += [f"train_label_count_{t}" for t in tags]
        fields += [f"accuracy_layer{i}" for i in layers] + [f"gain_vs_baseline_layer{i}" for i in layers]
        write_csv(out / f"word_types_{regime}.csv", fields,
                  ([w, int(n_train[j]), int(n_eval[j]), h[j], rows[j]["entropy_bits"],
                    tags[prediction[j]], baseline[j], not seen[j], *train[j], *accuracy[j], *gain[j]]
                   for j, w in enumerate(words)))
        for name, mask in (("all", np.ones(len(words), dtype=bool)), ("seen", seen), ("unseen", ~seen)):
            if mask.any():
                coverage.append([regime, name, int(mask.sum()), int(n_eval[mask].sum()),
                                 baseline[mask].mean(), accuracy[mask, 0].mean(), middle[mask].mean()])
        bins = [("H = 0", h == 0), ("0 < H <= 0.5", (h > 0) & (h <= 0.5)),
                ("0.5 < H <= 1", (h > 0.5) & (h <= 1))]
        if regime == "full":
            bins.append(("H > 1", h > 1))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        axes[1].axhline(0, color="0.5", linewidth=1)
        for threshold in sorted({1, 5, 10, 20, minimum}):
            selected = (n_train >= threshold) & (n_eval >= threshold)
            for name, values in [(f"gain_layer{i}", gain[:, i]) for i in layers] + [
                ("mean_layers4_8_minus_baseline", middle - baseline),
                ("mean_layers4_8_minus_layer0", middle - accuracy[:, 0])]:
                rho = p = np.nan
                # Equivalent arithmetic must not split tied ranks at machine precision.
                x, y = np.round(h[selected], 12), np.round(values[selected], 12)
                if selected.sum() > 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
                    rho, p = spearmanr(x, y)
                correlations.append([regime, threshold, int(selected.sum()), name, rho, p])
            for label, group in bins:
                mask = selected & group
                n = int(mask.sum())
                if not n:
                    continue
                mean, base = accuracy[mask].mean(axis=0), baseline[mask].mean()
                curves.extend([regime, threshold, label, n, int(n_eval[mask].sum()), int(i),
                               mean[i], base, mean[i] - base, mean[i] - mean[0]] for i in layers)
                if threshold == minimum:
                    table.append([regime, label, n, base, mean[0], mean[4:9].mean(), mean[4:9].mean() - base])
                    line, = axes[0].plot(layers, 100 * mean, marker=".", label=f"{label} (n={n:,})")
                    axes[0].axhline(100 * base, color=line.get_color(), linestyle="--", alpha=0.6)
                    axes[1].plot(layers, 100 * (mean - base), marker=".", color=line.get_color())
        for ax in axes:
            ax.set_xlabel("BERT layer (0 = embeddings)")
            ax.set_xticks(layers)
            ax.grid(alpha=0.15)
        axes[0].set_ylabel("Mean word-type accuracy (%)")
        axes[0].set_ylim(0, 100)
        axes[0].set_title("Solid: layer probe; dashed: training-majority baseline")
        axes[0].legend(fontsize=8, loc="lower right")
        axes[1].set_ylabel("Accuracy gain over baseline (percentage points)")
        axes[1].set_title("Does context improve on word identity alone?")
        fig.suptitle(f"{dataset} / {regime} / seed {seed} — training-label entropy\n"
                     f"Exact-spelling types with ≥{minimum} occurrences in both train and evaluation")
        for extension in ("png", "pdf"):
            fig.savefig(out / f"entropy_curves_{regime}.{extension}", dpi=180)
        plt.close(fig)
    write_csv(out / "curves.csv", ["regime", "min_count_each_split", "entropy_group", "types",
              "eval_occurrences", "layer", "accuracy", "baseline_accuracy", "gain_vs_baseline", "gain_vs_layer0"], curves)
    write_csv(out / "correlations.csv", ["regime", "min_count_each_split", "types", "metric", "spearman_rho", "p_two_sided"], correlations)
    write_csv(out / "coverage.csv", ["regime", "training_coverage", "types", "eval_occurrences",
              "baseline_accuracy", "layer0_accuracy", "mean_layers4_8_accuracy"], coverage)
    lines = [f"# {dataset}: layer curves by training-label entropy", "",
             f"Seed {seed}. No retraining or new inference: reuses saved word-type scores.",
             "Training counts use the probe's original dataset subset, truncation, and first-WordPiece word handling.",
             "Word identity is exact surface spelling. Entropy and baseline predictions use training labels ONLY.",
             "The baseline predicts each word's most frequent training label; ties choose the lowest label index.",
             "Binary baseline counts are collapsed BEFORE choosing the majority; neural predictions retain the existing full-label-then-collapse rule.",
             "Unseen words use the global training-majority label, have undefined training entropy, and are excluded from entropy curves/correlations.", "",
             f"Main curves require ≥{minimum} occurrences in EACH split; every qualifying word type has equal weight.",
             "All-layer ties are retained: these curves do not select a best layer. Dashed lines are each group's baseline accuracy.", "",
             "## Group averages", "",
             "| Regime | Training entropy | Types | Baseline % | Layer 0 % | Layers 4–8 mean % | Gain vs baseline (pp) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for regime, label, n, base, first, middle, delta in table:
        lines.append(f"| {regime} | {label} | {n:,} | {100*base:.2f} | {100*first:.2f} | {100*middle:.2f} | {100*delta:+.2f} |")
    lines += ["", "## Training-vocabulary coverage (all evaluated types)", "",
              "| Regime | Coverage | Types | Eval occurrences | Baseline % | Layer 0 % | Layers 4–8 mean % |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for regime, name, n, occurrences, base, first, middle in coverage:
        lines.append(f"| {regime} | {name} | {n:,} | {occurrences:,} | {100*base:.2f} | {100*first:.2f} | {100*middle:.2f} |")
    lines += ["", "## Exploratory Spearman correlations", "",
              "Training entropy versus mean accuracy across layers 4–8 minus the baseline (no per-word best-layer selection).", "",
              "Rank inputs are rounded to 12 decimal places to preserve numerical ties.", "",
              "| Regime | Minimum in each split | Types | Rho | Two-sided p |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for regime, threshold, n, name, rho, p in correlations:
        if name == "mean_layers4_8_minus_baseline":
            lines.append(f"| {regime} | {threshold} | {n:,} | {rho:.4f} | {p:.3g} |")
    lines += ["", "## Interpretation limits", "",
              "The 4–8 summary is exploratory, chosen after inspecting the earlier layer curves; it is not a preregistered test.",
              "Per-layer results and frequency thresholds are preserved in the CSVs rather than selecting whichever gives the largest correlation.",
              "A training-majority baseline can make mistakes even for zero-training-entropy words because labels can differ at evaluation.",
              "Positive gains show improvement over this word-only lookup, not causal proof of contextual disambiguation:",
              "BERT also provides pretrained lexical information, and baseline accuracy/headroom and word frequency remain confounders.",
              "P-values are asymptotic and unadjusted; they ignore shared sentences and other dependence. One seed; no uncertainty bands.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved %s", out / "summary.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-count", type=int, default=10)
    args = parser.parse_args()
    if args.min_count < 1:
        parser.error("--min-count must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for dataset in args.datasets:
        analyze(canonical_name(dataset), args.seed, args.min_count)


if __name__ == "__main__":
    main()
