"""STS-B sufficiency test, 3 panels.

  left    one encoder trained+evaluated on STS-B against its random-selection
          baseline -- does a trained selector beat random at equal budget?
  center  the same, every encoder that has an STS-B training run.
  right   cross-corpus generalisation: selectors trained on each tagging corpus,
          all evaluated on STS-B.

Each curve is the mean over the experiment's seed runs with a +-std band. The
source is every run's data/spearman_curves.json, written by
SelectorTrainer.final_eval() -> src/retrival_fun.py's run_stsb_sweep. Because
runtime.eval.stsb defaults to true, EVERY training run carries that file
regardless of what it trained on, which is what makes the right-hand panel free.

Discovery, not signatures. Earlier versions required a --panel*-sigs flag per
curve and named corpora (conll2000, movie_rationales) this store no longer has,
so the script could not run at all without hand-collected hex. It now reads the
store the way the other analysis scripts do and plots whatever exists; the flags
survive as overrides for narrowing.

    python3 utils/plot_stsb_sufficiency.py
    python3 utils/plot_stsb_sufficiency.py --panel1-family e5 --encoders sbert,e5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils._common import ENCODER_COLOR, ENCODER_DISPLAY  # noqa: E402

XPS = ROOT / "outputs/xps"
CURVES = "data/spearman_curves.json"

DATASET_DISPLAY = {
    "stsb": "STS-B", "wikiann": "WikiANN", "conll2003": "CoNLL-03",
    "fewnerd": "Few-NERD", "fewnerd_fine": "Few-NERD fine",
    "ud_upos": "UD UPOS", "ud_deprel": "UD deprel", "ud_discourse": "UD discourse",
    "conll2000": "CoNLL-00", "movie_rationales": "Movie Review",
}
DATASET_COLOR = {
    "stsb": "#0072B2", "wikiann": "#E69F00", "conll2003": "#009E73",
    "fewnerd": "#D55E00", "fewnerd_fine": "#CC79A7", "ud_upos": "#56B4E9",
    "ud_deprel": "#8C564B", "ud_discourse": "#7F7F7F",
    "conll2000": "#BCBD22", "movie_rationales": "#E377C2",
}
PANEL3_ORDER = ["stsb", "wikiann", "conll2003", "fewnerd", "fewnerd_fine",
                "ud_upos", "ud_deprel", "ud_discourse"]


def discover(pooling: str | None = "mean") -> dict[tuple[str, str], list[Path]]:
    """(dataset, family) -> run dirs that carry an STS-B curve."""
    found: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for cfg_path in sorted(XPS.glob("*/config.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text())
        enc = cfg["data"]["encoder"]
        if pooling is not None and enc.get("pooling") != pooling:
            continue
        runs = [r.parent.parent for r in cfg_path.parent.glob(f"*/{CURVES}")]
        if runs:
            found[(cfg["data"]["dataset"], enc["family"])].extend(runs)
    return found


def load(paths: list[Path]):
    """(rho, selector[S,R], random[S,R]) stacked over the seed runs."""
    rho = sel = rnd = None
    sels, rnds = [], []
    for p in paths:
        payload = json.loads((p / CURVES).read_text())
        rho = np.array(payload["rho"], dtype=float)
        sels.append(np.array(payload["curves"]["selector"], dtype=float))
        rnds.append(np.array(payload["curves"]["random"], dtype=float))
    return rho, np.stack(sels), np.stack(rnds)


def band(ax, x, ys, label, color=None, linestyle="-", marker="o", alpha=0.18):
    m, s = ys.mean(0), ys.std(0)
    line, = ax.plot(x, m, marker=marker, ms=4, lw=2.0, ls=linestyle, label=label, color=color)
    ax.fill_between(x, m - s, m + s, alpha=alpha, color=line.get_color())
    return line.get_color()


def style(ax):
    ax.grid(True, color="#DDDDDD", lw=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_color("#BBBBBB")
    ax.set_xlabel("ρ")
    ax.set_ylabel("spearman")


def main(a):
    found = discover(None if a.pooling == "any" else a.pooling)
    stsb = {fam: runs for (ds, fam), runs in found.items() if ds == "stsb"}
    encoders = ([e for e in a.encoders.split(",") if e in stsb] if a.encoders
                else [e for e in ENCODER_DISPLAY if e in stsb])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    notes = []

    # --- panel 1: one encoder, trained + evaluated on STS-B, vs random
    style(ax1)
    fam1 = a.panel1_family if a.panel1_family in stsb else (encoders[0] if encoders else None)
    if fam1:
        rho, sel, rnd = load(stsb[fam1])
        band(ax1, rho, sel, ENCODER_DISPLAY[fam1], color=ENCODER_COLOR[fam1])
        band(ax1, rho, rnd, "random", color="#999999", linestyle="--", marker="x")
        ax1.set_title(f"STS-B sufficiency — {ENCODER_DISPLAY[fam1]}  ({len(stsb[fam1])} seeds)")
        ax1.legend(frameon=False, fontsize=8)
    else:
        ax1.set_title("STS-B sufficiency — no STS-B runs yet")
        ax1.text(.5, .5, "run:  forge grid --file utils/grid_stsb.yaml",
                 ha="center", va="center", fontsize=9, transform=ax1.transAxes)
        notes.append("panel 1/2 empty: no experiment has data.dataset=stsb")

    # --- panel 2: every encoder trained + evaluated on STS-B
    style(ax2)
    for fam in encoders:
        rho, sel, rnd = load(stsb[fam])
        band(ax2, rho, sel, ENCODER_DISPLAY[fam], color=ENCODER_COLOR[fam])
    if encoders:
        rho, _, rnd = load(stsb[encoders[0]])
        band(ax2, rho, rnd, "random", color="#999999", linestyle="--", marker="x")
        ax2.legend(frameon=False, fontsize=7, ncol=2)
    ax2.set_title("Encoders trained on STS-B")

    # --- panel 3: cross-corpus, selectors trained elsewhere, evaluated on STS-B
    style(ax3)
    fam3 = a.panel3_family
    drawn = 0
    for ds in PANEL3_ORDER:
        runs = found.get((ds, fam3))
        if not runs:
            continue
        rho, sel, rnd = load(runs)
        band(ax3, rho, sel, f"{DATASET_DISPLAY[ds]} ({len(runs)})", color=DATASET_COLOR[ds])
        if drawn == 0:
            # One shared random baseline: random selection does not depend on
            # which corpus the selector was trained on.
            band(ax3, rho, rnd, "random", color="#999999", linestyle="--", marker="x")
        drawn += 1
    ax3.set_title(f"{ENCODER_DISPLAY[fam3]} trained elsewhere, tested on STS-B")
    ax3.legend(frameon=False, fontsize=7, ncol=2)
    if drawn == 0:
        notes.append(f"panel 3 empty: no {fam3} runs at pooling={a.pooling}")

    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=160)
    print(f"saved {a.output}")
    print(f"discovered {sum(len(v) for v in found.values())} runs carrying {CURVES} "
          f"across {len(found)} (dataset, encoder) pairs")
    for (ds, fam), runs in sorted(found.items()):
        print(f"   {ds:14s}{fam:9s}{len(runs)} run(s)")
    for n in notes:
        print(f"NOTE: {n}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pooling", default="mean", help='pooling to keep, or "any" (default: mean)')
    p.add_argument("--encoders", default="", help="comma-separated subset for panel 2")
    p.add_argument("--panel1-family", default="sbert", help="encoder for panel 1 (default: sbert)")
    p.add_argument("--panel3-family", default="sbert", help="encoder for panel 3 (default: sbert)")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/analysis/stsb/stsb_sufficiency.png")
    main(p.parse_args())
