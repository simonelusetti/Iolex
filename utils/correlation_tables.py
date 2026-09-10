"""Correlation of every selection metric against tagger F1.

The unit is the TAG: one number per (series, metric, variant, seed, rho), each
a Pearson correlation across the dataset's tags.

    x  the series' per-tag selection bias  (see _common.load_bias)
         selection = observed rate - rho          division = signed z
         signed    = as measured                  absolute = |value|, per run
    y  that tag's token-level F1 from the tagger cache, averaged over the
       tagger's seeds
    r  pearson(x, y) ACROSS TAGS

Tag count is the binding constraint on everything here. At wikiann's 6 tags
|r| must exceed 0.811 to reach p<0.05; at UD deprel's 29 it is 0.367. Every
table prints its own critical value, because a cell means nothing without it.

Driven by utils/analysis.yaml -- see analysis.py.
Importable directly too: build_report() returns plain dicts.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge.core import ExperimentStore  # noqa: E402
from utils._common import entity_tags, load_bias, probe_reports, tagger_f1  # noqa: E402
from utils.plot_grounding import discover_series  # noqa: E402

DEFAULT_RHOS = [None] + [round(0.1 * i, 1) for i in range(1, 10)]
DEFAULT_METRICS = ["selection", "division"]
VARIANTS = {"signed": False, "absolute": True}
SHORT = {"signed": "sgn", "absolute": "abs"}


def critical_r(n: int, alpha: float = 0.05) -> float:
    """|r| needed for p<alpha at n points, for ONE correlation.

    This is the threshold for a single cell and nothing else. It is the wrong
    bar for the mean over seeds -- see permutation_p, which is the test that
    question actually needs.
    """
    if n < 3:
        return float("nan")
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    return float(np.sqrt(t**2 / (t**2 + n - 2)))


def _corr_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson r between every row of X [S,n] and every row of Y [P,n] -> [S,P]."""
    Xc = X - X.mean(1, keepdims=True)
    Yc = Y - Y.mean(1, keepdims=True)
    Xn = np.linalg.norm(Xc, axis=1, keepdims=True)
    Yn = np.linalg.norm(Yc, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Xc / Xn) @ (Yc / Yn).T


def permutation_matrix(n: int, draws: int, max_exact: int, seed: int = 0):
    """Every relabelling of the tags when n! is small, else `draws` random ones."""
    if math.factorial(n) <= max_exact:
        return np.array(list(itertools.permutations(range(n)))), True
    rng = np.random.default_rng(seed)
    return np.array([rng.permutation(n) for _ in range(draws)]), False


def permutation_p(xs: np.ndarray, y: np.ndarray, perms: np.ndarray) -> tuple[float, float]:
    """(mean r over seeds, two-sided p, critical |mean r|) for the aggregate claim.

    The question "all my seeds agree on a weak r, is that anything?" is not
    answered by the single-cell critical value: the mean of k correlations is
    less variable than one of them, so that bar is far too high for it.

    Tested by relabelling the tags. Under the null that per-tag bias and
    per-tag F1 are unrelated, which tag carries which F1 is arbitrary, so the
    null distribution of the mean r is obtained by permuting y and recomputing.

    The SAME permutation is applied to every seed, which is what makes this
    valid here: seeds share one y and one encoder and are emphatically not
    independent replicates, but permuting them together leaves whatever
    dependence they have untouched. No normality assumption either, and with
    7 tags all 5040 relabellings are enumerable, so the p-value is exact.
    """
    X = np.atleast_2d(xs)
    if not np.isfinite(X).all() or X.std(axis=1).min() == 0 or y.std() == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float(_corr_rows(X, y[None, :])[:, 0].mean())
    null = _corr_rows(X, y[perms]).mean(axis=0)
    p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    # The bar that |mean r| actually has to clear here, directly comparable to
    # the single-cell critical value and normally well below it.
    crit = float(np.quantile(np.abs(null), 0.95))
    return obs, p, crit


def sign_consistency(values: list[float]) -> tuple[int, int, float]:
    """(agreeing, total, two-sided sign-test p) -- the plainest form of the claim.

    Weaker than the permutation test and it does assume seeds are independent,
    which they are not, so it is reported alongside rather than instead.
    """
    vals = [v for v in values if np.isfinite(v) and v != 0]
    if not vals:
        return 0, 0, float("nan")
    pos = sum(1 for v in vals if v > 0)
    agree = max(pos, len(vals) - pos)
    return agree, len(vals), float(min(1.0, 2 * stats.binom.sf(agree - 1, len(vals), 0.5)))


def _seed_of(store: ExperimentStore, path: Path):
    return OmegaConf.select(store.load_run(path.parent.name, path.name).config, "seed")


def _rho_label(rho) -> str:
    return "avg" if rho is None else f"{rho:.1f}"


def available_datasets() -> list[str]:
    """Every dataset with a selector or oracle run in the store."""
    store = ExperimentStore(root=ROOT / "outputs")
    found = set()
    for selection in store.all_selections():
        cfg = selection.experiment.config
        if str(OmegaConf.select(cfg, "task")) in ("rationale", "oracle"):
            if any(r.status == "done" for r in (selection.runs or [])):
                found.add(str(OmegaConf.select(cfg, "data.dataset")))
    return sorted(found)


def _listify(value, default):
    """null means "everything available"; anything else narrows to what it names.

    A bare string is one name, or several if comma-separated -- not a sequence
    of characters, which is what list("bert/mean") would give. Hydra passes a
    plain str whenever the override omits brackets (`series=bert/mean`), which
    is the most natural thing to type.
    """
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value)


def build_report(cfg) -> dict:
    """Every requested correlation, as plain data, for every requested dataset.

    Structure mirrors how the numbers are read: report["datasets"][name]
    ["series"][label]["cells"] is a list of {metric, variant, rho, seed, r}, so
    a consumer filters instead of re-deriving anything.
    """
    a = cfg.analysis if "analysis" in cfg else cfg
    datasets = _listify(a.datasets, available_datasets())
    if not datasets:
        raise SystemExit("no dataset has selector or oracle runs in the store")
    out = {"datasets": {}, "skipped": {}}
    for name in datasets:
        try:
            out["datasets"][name] = build_dataset_report(a, name)
        except SystemExit as exc:      # too few tags, no runs: note and continue
            if len(datasets) == 1:
                raise
            out["skipped"][name] = str(exc)
    out["n_cells"] = sum(s["n_cells"] for s in out["datasets"].values())
    perm_cfg = a.permutation if "permutation" in a else None
    if out["datasets"]:
        out["global"] = build_global(
            out, draws=int(getattr(perm_cfg, "draws", 20000) or 20000) if perm_cfg else 20000)
    return out


def build_dataset_report(a, dataset: str) -> dict:
    # null = exclude nothing, consistent with the null-means-everything rule.
    exclude = [] if a.labels.exclude is None else list(a.labels.exclude)
    tags = entity_tags(dataset, exclude)
    if len(tags) < 3:
        raise SystemExit(
            f"{dataset} has {len(tags)} tags after excluding {exclude}; "
            "a correlation needs at least 3.")

    rhos = [None if r in (None, "avg") else float(r) for r in _listify(a.rhos, DEFAULT_RHOS)]
    metrics = _listify(a.metrics, DEFAULT_METRICS)
    variants = _listify(a.variants, VARIANTS)
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; expected {sorted(VARIANTS)}")

    perm_cfg = a.permutation if "permutation" in a else None
    draws = int(getattr(perm_cfg, "draws", 20000) or 20000) if perm_cfg else 20000
    max_exact = int(getattr(perm_cfg, "max_exact", 50400) or 50400) if perm_cfg else 50400
    perms, exact = permutation_matrix(len(tags), draws, max_exact)

    store = ExperimentStore(root=ROOT / "outputs")
    found = {s["label"]: s for s in discover_series(dataset)}
    available = sorted(found)
    absent: list[str] = []
    if a.series is not None:
        # Narrow to what this dataset actually has. A series naming something
        # absent here is normal, not an error: oracles exist only for some
        # corpora, so [bert/mean, bert/mean/oracle] should still report
        # bert/mean everywhere rather than discarding four datasets to
        # complain about the one series they lack.
        wanted = _listify(a.series, [])
        absent = [w for w in wanted if w not in found]
        found = {k: v for k, v in found.items() if k in wanted}
    if not found:
        if a.series is None:
            raise SystemExit(f"no selector or oracle runs found for dataset {dataset!r}")
        raise SystemExit(
            f"none of {absent} exist for dataset {dataset!r}; "
            f"it has {sorted(available)}")

    report = {
        "dataset": dataset,
        "tags": tags,
        "excluded_labels": exclude,
        "n_tags": len(tags),
        "critical_r_p05": critical_r(len(tags)),
        "n_permutations": int(len(perms)),
        "permutations_exact": bool(exact),
        "per_seed": bool(a.per_seed),
        "rhos": [_rho_label(r) for r in rhos],
        "series": {},
        "absent_series": absent,
    }
    for label in sorted(found):
        entry = found[label]
        reports = probe_reports(dataset, entry["family"])
        f1 = np.array([np.mean([tagger_f1(r, tags)[g] for r in reports]) for g in tags])
        runs = sorted(entry["bias_runs"],
                      key=lambda p: (_seed_of(store, p) is None, _seed_of(store, p)))
        groups = ([("pooled", runs)] if not a.per_seed
                  else [(_seed_of(store, p), [p]) for p in runs])

        cells, agg = [], []
        by_cell: dict[tuple, list[np.ndarray]] = {}
        for metric in metrics:
            for variant in variants:
                absolute = VARIANTS[variant]
                for seed, paths in groups:
                    for rho in rhos:
                        biases = [load_bias(p, dataset, metric, rho) for p in paths]
                        x = np.array([np.mean([abs(b[g]) if absolute else b[g] for b in biases])
                                      for g in tags])
                        r = float(stats.pearsonr(x, f1)[0]) if x.std() > 0 else float("nan")
                        # The per-tag x behind the r, carried so a plot (or a
                        # re-read of the JSON months later) can show the
                        # scatter without re-deriving anything from the store.
                        cells.append({"metric": metric, "variant": variant,
                                      "rho": _rho_label(rho), "seed": str(seed), "r": r,
                                      "x": {g: float(v) for g, v in zip(tags, x)}})
                        by_cell.setdefault((metric, variant, _rho_label(rho)), []).append(x)
        for (metric, variant, rho_label), xs in by_cell.items():
            r_mean, p_perm, crit_mean = permutation_p(np.array(xs), f1, perms)
            rs = [c["r"] for c in cells if c["metric"] == metric
                  and c["variant"] == variant and c["rho"] == rho_label]
            agree, total, p_sign = sign_consistency(rs)
            agg.append({"metric": metric, "variant": variant, "rho": rho_label,
                        "n_seeds": len(xs), "r_mean": r_mean, "p_perm": p_perm,
                        "critical_r_mean": crit_mean,
                        "sign_agree": agree, "sign_n": total, "p_sign": p_sign})

        report["series"][label] = {
            "signature": entry["signature"],
            "family": entry["family"],
            "pooling": entry["pooling"],
            "task": entry["task"],
            "n_bias_runs": len(runs),
            "n_tagger_seeds": len(reports),
            "tagger_f1": {g: float(v) for g, v in zip(tags, f1)},
            "cells": cells,
            "aggregate": agg,
        }
    report["n_cells"] = sum(len(s["cells"]) for s in report["series"].values())
    return report


def build_global(report: dict, draws: int = 20000, seed: int = 0) -> dict:
    """One test per (metric, variant, rho) pooling every experiment in the store.

    Worth doing precisely because it is the one axis that adds information.
    Seeds of one experiment correlate at ~0.98 and pooling them moves the bar
    by 0.004; datasets are genuinely independent, so combining them does buy
    power.

    Construction: within a dataset, the statistic is the mean r over all its
    series and seeds; globally it is the unweighted mean of those per-dataset
    numbers, so a corpus with many series does not outvote one with few. The
    null draws an INDEPENDENT tag relabelling per dataset and averages the
    same way, which is legitimate exactly because the corpora are unrelated --
    and is why the global test is not just the per-dataset ones restated.

    Monte Carlo rather than exact: the datasets have different tag counts, so
    there is no shared enumeration to run over.
    """
    rng = np.random.default_rng(seed)
    per_ds: dict[str, dict] = {}
    keys: set[tuple[str, str, str]] = set()

    for name, dsrep in report["datasets"].items():
        tags = dsrep["tags"]
        perms = np.array([rng.permutation(len(tags)) for _ in range(draws)])
        # y is per SERIES, not per dataset: each family is scored against its
        # own probe, so sbert/mean's F1 vector is not bert/mean's. Pooling
        # them against one y would correlate half the runs with the wrong
        # target. The permutation is shared -- it relabels tags, and every
        # series is indexed by the same tag list.
        f1_of = {lab: np.array([s["tagger_f1"][g] for g in tags])
                 for lab, s in dsrep["series"].items()}
        cells: dict[tuple[str, str, str], dict[str, list[list[float]]]] = {}
        for lab, s in dsrep["series"].items():
            for c in s["cells"]:
                key = (c["metric"], c["variant"], c["rho"])
                cells.setdefault(key, {}).setdefault(lab, []).append(
                    [c["x"][g] for g in tags])

        entry = {}
        for key, by_label in cells.items():
            obs_sum, null_sum, n = 0.0, np.zeros(draws), 0
            ok = True
            for lab, xs in by_label.items():
                X = np.array(xs)
                if not np.isfinite(X).all() or X.std(axis=1).min() == 0:
                    ok = False
                    break
                y = f1_of[lab]
                obs_sum += float(_corr_rows(X, y[None, :])[:, 0].sum())
                null_sum += _corr_rows(X, y[perms]).sum(axis=0)
                n += len(xs)
            if not ok or n == 0:
                continue
            entry[key] = {"obs": obs_sum / n, "null": null_sum / n, "n": n}
            keys.add(key)
        per_ds[name] = entry

    out = []
    for metric, variant, rho in sorted(keys):
        got = [(n, e[(metric, variant, rho)]) for n, e in per_ds.items()
               if (metric, variant, rho) in e]
        obs = float(np.mean([g["obs"] for _, g in got]))
        null = np.mean(np.stack([g["null"] for _, g in got]), axis=0)
        out.append({
            "metric": metric, "variant": variant, "rho": rho,
            "n_datasets": len(got), "n_cells": sum(g["n"] for _, g in got),
            "r_mean": obs,
            "p_perm": float((np.abs(null) >= abs(obs) - 1e-12).mean()),
            "critical_r_mean": float(np.quantile(np.abs(null), 0.95)),
            "per_dataset": {n: float(g["obs"]) for n, g in got},
        })
    return {"draws": int(draws), "rows": out,
            "datasets": sorted(per_ds), "exact": False}


def render_global(glob: dict, out) -> None:
    p = lambda *a: print(*a, file=out)
    names = glob["datasets"]
    p("=" * 78)
    p(f"GLOBAL   every experiment pooled over {len(names)} dataset(s): "
      f"{', '.join(names)}")
    p(f"  mean r per dataset, then unweighted across datasets; null relabels "
      f"each\n  dataset's tags independently, {glob['draws']:,} draws (Monte Carlo)")
    p("=" * 78)
    head = (f"{'metric/variant':18s}{'rho':>5s}{'cells':>7s}{'mean r':>9s}"
            f"{'crit':>8s}{'p':>9s}   per-dataset")
    p(head)
    for row in glob["rows"]:
        per = "  ".join(f"{n.split('_')[-1][:6]}={v:+.2f}"
                        for n, v in row["per_dataset"].items())
        star = " *" if row["p_perm"] < 0.05 else "  "
        p(f"{row['metric'] + '/' + SHORT[row['variant']]:18s}{row['rho']:>5s}"
          f"{row['n_cells']:>7d}{row['r_mean']:>+9.3f}{row['critical_r_mean']:>8.3f}"
          f"{row['p_perm']:>9.4f}{star} {per}")


def render_text(report: dict, out) -> None:
    for name, sub in report["datasets"].items():
        render_dataset(sub, out)
        print(file=out)
    for name, why in report.get("skipped", {}).items():
        print(f"skipped {name}: {why}", file=out)
    if report.get("global"):
        print(file=out)
        render_global(report["global"], out)


def render_dataset(report: dict, out) -> None:
    rhos = report["rhos"]
    head = f"{'metric/variant':18s}{'seed':>7s}" + "".join(f"{r:>8s}" for r in rhos)
    p = lambda *a: print(*a, file=out)
    p(f"dataset={report['dataset']}   {report['n_tags']} tags "
      f"(excluded: {report['excluded_labels'] if report['excluded_labels'] else 'none'})")
    p(f"  {', '.join(report['tags'])}")
    p(f"|r| for p<0.05 at n={report['n_tags']}: {report['critical_r_p05']:.3f}"
      f"   mode: {'per seed' if report['per_seed'] else 'POOLED over seeds'}"
      f"   cells: {report['n_cells']}")
    if report.get("absent_series"):
        # Reported, not fatal: a series this corpus lacks is expected.
        p(f"  (not present here: {', '.join(report['absent_series'])})")
    for label, s in report["series"].items():
        p("\n" + "=" * len(head))
        p(f"{label}   {s['signature']}   {s['n_bias_runs']} run(s), "
          f"tagger seeds={s['n_tagger_seeds']}")
        p(head)
        seen = None
        by = {}
        for c in s["cells"]:
            by.setdefault((c["metric"], c["variant"], c["seed"]), {})[c["rho"]] = c["r"]
        for (metric, variant, seed), row in by.items():
            name = f"{metric}/{SHORT[variant]}"
            first = name != seen
            seen = name
            p(f"{name if first else '':18s}{seed:>7s}"
              + "".join(f"{row.get(r, float('nan')):+8.3f}" for r in rhos))

        if s.get("aggregate"):
            _render_aggregate(s["aggregate"], rhos, p)


def _render_aggregate(agg: list[dict], rhos: list[str], p) -> None:
    """Mean r over seeds and its permutation p -- the test for "they all agree".

    Printed under each series rather than in place of the per-seed rows: the
    per-seed spread is what shows whether the mean is a real consensus or an
    average of contradictions, and the p alone would hide that.
    """
    by = {}
    for c in agg:
        by.setdefault((c["metric"], c["variant"]), {})[c["rho"]] = c
    p(f"{'-' * (25 + 8 * len(rhos))}")
    seen = None
    for (metric, variant), row in by.items():
        name = f"{metric}/{SHORT[variant]}"
        first = name != seen
        seen = name
        p(f"{name if first else '':18s}{'mean':>7s}"
          + "".join(f"{row[r]['r_mean']:+8.3f}" if r in row else f"{'':>8s}"
                    for r in rhos))
        p(f"{'':18s}{'p perm':>7s}"
          + "".join(f"{row[r]['p_perm']:>8.4f}" if r in row else f"{'':>8s}"
                    for r in rhos))
        p(f"{'':18s}{'crit':>7s}"
          + "".join(f"{row[r]['critical_r_mean']:>8.3f}" if r in row else f"{'':>8s}"
                    for r in rhos))
        p(f"{'':18s}{'signs':>7s}"
          + "".join(f"{str(row[r]['sign_agree']) + '/' + str(row[r]['sign_n']):>8s}"
                    if r in row else f"{'':>8s}" for r in rhos))
