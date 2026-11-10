"""Evaluation reporting entry point.

    A = "--config-dir utils --config-name analysis"
    forge $A -M analysis run
    forge $A -M analysis run dataset=ud_upos labels.exclude=[]
    forge $A -M analysis run metrics=[division] per_seed=false
    forge $A -M analysis run output.json=outputs/analysis/wikiann.json

Deliberately NOT a forge experiment: it calls no start_run and writes nothing
into outputs/xps. An analysis reads finished runs; it never contributes to one.

Its settings live in their own file, utils/analysis.yaml, rather than in
conf/config.yaml. That separation is the point: every non-excluded key in
conf/config.yaml is part of an experiment's identity, so a reporting option
living there could fork experiments that never read it. Nothing here can.

What it uses forge for is the config machinery -- composition plus dotted CLI
overrides -- so a report's settings are stated in one place and recorded next to
its numbers instead of living in shell history. Defaults are broad (every
metric, variant and rho), so a bare run is a complete report and narrowing is
how a question gets isolated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.correlation_plots import save_plots  # noqa: E402
from utils.correlation_tables import build_report, render_text  # noqa: E402


def main(cfg) -> int:
    report = build_report(cfg)
    # The exact settings that produced these numbers, carried with them.
    report["config"] = OmegaConf.to_container(cfg, resolve=True)

    out_path = cfg.output.path
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            render_text(report, fh)
        print(f"wrote {path}")
    else:
        render_text(report, sys.stdout)

    json_path = cfg.output.json
    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {path}")

    # Same report object the table just rendered, so a figure and its table
    # cannot disagree about what was measured.
    plots_dir = cfg.output.plots
    if plots_dir:
        for written in save_plots(report, plots_dir, kinds=cfg.plots.kinds,
                                  rho=str(cfg.plots.rho), spread=str(cfg.plots.spread)):
            print(f"wrote {written}")
    return 0
