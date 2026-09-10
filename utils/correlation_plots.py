"""Plots of the correlation report, shared across experiments.

Consumes exactly what utils/correlation_tables.build_report() returns -- the
same discovery, the same filtering, the same numbers the table prints. Nothing
here re-reads the store, so a figure and its table can never disagree.

Three views, each answering a different question:

  r_vs_rho    per dataset, r against rho, one line per series. Does the
              correlation hold across the budget, or only at one rho?
  summary     series x dataset at a single rho, one panel per metric/variant.
              The cross-experiment view: does a series behave the same way
              everywhere, or is the effect corpus-specific?
  scatter     the raw (bias, F1) points across tags behind one cell. The
              sanity check -- with 6 tags an r is one outlier away from
              anything, and only the scatter shows that.

Colour means encoder throughout (utils/_common.ENCODER_COLOR); the oracle is
drawn dashed so a selector and its ceiling stay distinguishable at a glance.

Driven by utils/analysis.yaml -- see analysis.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils._common import (  # noqa: E402
    ENCODER_COLOR, ENCODER_MARKER, make_flat_grey_cmap, make_flat_grey_norm,
    mean_spread,
)

KINDS = ("r_vs_rho", "summary", "scatter")


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

def series_style(series: dict) -> dict:
    """Colour by encoder, dash by task, marker by encoder. Fixed, never cycled."""
    family = series.get("family", "")
    return {
        "color": ENCODER_COLOR.get(family, "#666666"),
        "marker": ENCODER_MARKER.get(family, "o"),
        "linestyle": "--" if series.get("task") == "oracle" else "-",
    }


def _cells_by(series: dict, metric: str, variant: str) -> dict[str, list[float]]:
    """{rho label: [r over seeds]} for one metric/variant of one series."""
    out: dict[str, list[float]] = {}
    for c in series["cells"]:
        if c["metric"] == metric and c["variant"] == variant:
            out.setdefault(c["rho"], []).append(c["r"])
    return out


def _combos(report: dict) -> list[tuple[str, str]]:
    """(metric, variant) pairs present, in a stable order."""
    seen = []
    for s in report["series"].values():
        for c in s["cells"]:
            key = (c["metric"], c["variant"])
            if key not in seen:
                seen.append(key)
    return sorted(seen)


def _numeric_rhos(labels) -> list[str]:
    """Rho labels that are actually numbers -- "avg" is a summary, not a point."""
    return [r for r in labels if r != "avg"]


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# r against rho
# ---------------------------------------------------------------------------

def plot_r_vs_rho(dsrep: dict, outdir: Path, spread: str = "sd") -> Path | None:
    """One panel per (metric, variant); x = rho, y = r, one line per series.

    The shaded band is +/- the critical |r| for this dataset's tag count. A
    line inside it is indistinguishable from no correlation, which at 6 tags
    is most of the plot -- drawing the band is what stops a reader taking a
    large-looking r seriously.
    """
    rhos = _numeric_rhos(dsrep["rhos"])
    combos = _combos(dsrep)
    if not rhos or not combos:
        return None

    crit = dsrep["critical_r_p05"]
    x = [float(r) for r in rhos]
    fig, axes = plt.subplots(1, len(combos), figsize=(4.2 * len(combos), 3.8),
                             squeeze=False, sharey=True)
    for ax, (metric, variant) in zip(axes[0], combos):
        ax.axhspan(-crit, crit, color="#d9d9d9", zorder=0)
        ax.axhline(0.0, color="#888888", lw=0.8, zorder=1)
        for label, s in dsrep["series"].items():
            by = _cells_by(s, metric, variant)
            pairs = [(xi, by[r]) for xi, r in zip(x, rhos) if r in by]
            if not pairs:
                continue
            xs = [p[0] for p in pairs]
            stats_ = [mean_spread(p[1], spread) for p in pairs]
            mean = np.array([m for m, _ in stats_])
            rad = np.array([d for _, d in stats_])
            st = series_style(s)
            ax.plot(xs, mean, label=label, lw=1.6, ms=4, **st)
            if rad.any():
                ax.fill_between(xs, mean - rad, mean + rad,
                                color=st["color"], alpha=0.15, lw=0)
        ax.set_title(f"{metric} / {variant}", fontsize=10)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylim(-1.05, 1.05)
    axes[0][0].set_ylabel("pearson r  (bias vs tagger F1)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, max(1, len(labels))),
               frameon=False, bbox_to_anchor=(0.5, -0.12), fontsize=9)
    fig.suptitle(f"{dsrep['dataset']}   {dsrep['n_tags']} tags   "
                 f"grey = |r| < {crit:.3f} (p>0.05)", fontsize=11)
    return _save(fig, outdir / f"r_vs_rho_{dsrep['dataset']}.pdf")


# ---------------------------------------------------------------------------
# series x dataset
# ---------------------------------------------------------------------------

def plot_summary(report: dict, outdir: Path, rho: str = "avg") -> list[Path]:
    """series x dataset heatmap of r, one figure per (metric, variant).

    This is the cross-experiment view the per-dataset tables cannot give: the
    same series in a row across every corpus it was run on.

    Each dataset has its OWN critical r (it depends on tag count), so the
    grey "not significant" band cannot be a single colour scale across
    columns. The cell text carries the value and a marker for significance
    instead, and the colour is only a reading aid.
    """
    names = list(report["datasets"])
    if not names:
        return []
    labels = sorted({lab for n in names for lab in report["datasets"][n]["series"]})
    combos = sorted({c for n in names for c in _combos(report["datasets"][n])})
    written = []

    for metric, variant in combos:
        grid = np.full((len(labels), len(names)), np.nan)
        sig = np.zeros_like(grid, dtype=bool)
        for j, name in enumerate(names):
            dsrep = report["datasets"][name]
            crit = dsrep["critical_r_p05"]
            for i, lab in enumerate(labels):
                s = dsrep["series"].get(lab)
                if s is None:
                    continue
                vals = _cells_by(s, metric, variant).get(rho)
                if not vals:
                    continue
                grid[i, j] = float(np.mean(vals))
                sig[i, j] = abs(grid[i, j]) > crit

        fig, ax = plt.subplots(figsize=(1.5 + 1.35 * len(names), 1.2 + 0.5 * len(labels)))
        ax.imshow(grid, cmap=make_flat_grey_cmap(),
                  norm=make_flat_grey_norm(1.0, 0.0), aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        for i in range(len(labels)):
            for j in range(len(names)):
                if np.isnan(grid[i, j]):
                    # Painted over rather than left to the colormap: zero is
                    # mid-grey here, so an absent series and a measured null
                    # are the same colour. Neither set_bad nor a masked array
                    # can fix that, because this norm's forward calls
                    # np.asarray and drops the mask before matplotlib sees it.
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor="white", edgecolor="#eeeeee",
                                           zorder=2))
                    ax.text(j, i, "no run", ha="center", va="center", zorder=3,
                            fontsize=7, color="#bbbbbb")
                else:
                    ax.text(j, i, f"{grid[i, j]:+.2f}" + ("*" if sig[i, j] else ""),
                            ha="center", va="center", fontsize=8,
                            color="white" if abs(grid[i, j]) > 0.55 else "black")
        crits = ", ".join(f"{n} {report['datasets'][n]['critical_r_p05']:.2f}" for n in names)
        ax.set_title(f"{metric} / {variant}   rho={rho}\n* |r| over that corpus' "
                     f"critical value  ({crits})", fontsize=9)
        written.append(_save(fig, outdir / f"summary_{metric}_{variant}_rho{rho}.pdf"))
    return written


# ---------------------------------------------------------------------------
# the points behind one r
# ---------------------------------------------------------------------------

def plot_scatter(dsrep: dict, outdir: Path, metric: str, variant: str,
                 rho: str = "avg") -> Path | None:
    """Per-tag bias against tagger F1, the raw points behind one column.

    Seeds are averaged here purely to have one point per tag to draw; the r
    quoted in the table is still the per-seed one.
    """
    f1_by_series = {lab: s["tagger_f1"] for lab, s in dsrep["series"].items()}
    if not f1_by_series:
        return None
    tags = dsrep["tags"]
    series = list(dsrep["series"].items())
    ncol = min(3, len(series))
    nrow = int(np.ceil(len(series) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow),
                             squeeze=False)
    drew = False
    for ax, (label, s) in zip(axes.flat, series):
        xs = [c["x"] for c in s["cells"]
              if c["metric"] == metric and c["variant"] == variant and c["rho"] == rho]
        if not xs:
            ax.set_visible(False)
            continue
        drew = True
        x = np.array([np.mean([d[g] for d in xs]) for g in tags])
        y = np.array([s["tagger_f1"][g] for g in tags])
        st = series_style(s)
        ax.scatter(x, y, color=st["color"], marker=st["marker"], s=42, zorder=3)
        # Room for the tag labels: without it the extreme point -- usually O,
        # which is exactly the one worth reading -- has its name clipped by
        # the axes edge.
        ax.margins(0.18)
        for g, xi, yi in zip(tags, x, y):
            ax.annotate(g, (xi, yi), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        if len(tags) > 2 and x.std() > 0:
            b, a = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 10)
            ax.plot(xx, a + b * xx, color=st["color"], lw=1.0, alpha=0.6)
        rs = [c["r"] for c in s["cells"]
              if c["metric"] == metric and c["variant"] == variant and c["rho"] == rho]
        ax.set_title(f"{label}\nr={np.mean(rs):+.3f} over {len(rs)} seed(s)", fontsize=9)
        ax.set_xlabel(f"{metric}/{variant}")
        ax.set_ylabel("tagger F1")
    for ax in axes.flat[len(series):]:
        ax.set_visible(False)
    if not drew:
        plt.close(fig)
        return None
    fig.suptitle(f"{dsrep['dataset']}   rho={rho}   "
                 f"critical |r|={dsrep['critical_r_p05']:.3f}", fontsize=11)
    fig.tight_layout()
    return _save(fig, outdir / f"scatter_{dsrep['dataset']}_{metric}_{variant}_rho{rho}.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def save_plots(report: dict, outdir, kinds=None, rho: str = "avg",
               spread: str = "sd") -> list[Path]:
    """Every requested view of `report`. Returns what was written."""
    outdir = Path(outdir)
    kinds = list(KINDS) if kinds is None else [str(k) for k in kinds]
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        raise SystemExit(f"unknown plot kind(s) {unknown}; expected {list(KINDS)}")

    written: list[Path] = []
    if "summary" in kinds:
        written += plot_summary(report, outdir, rho)
    for dsrep in report["datasets"].values():
        if "r_vs_rho" in kinds:
            p = plot_r_vs_rho(dsrep, outdir, spread)
            if p:
                written.append(p)
        if "scatter" in kinds:
            for metric, variant in _combos(dsrep):
                p = plot_scatter(dsrep, outdir, metric, variant, rho)
                if p:
                    written.append(p)
    return written
