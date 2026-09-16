"""Write data/selection_log.npz into finished selector runs that predate it.

The per-word record is a by-product of evaluation, so producing it for an old
run costs one evaluation pass -- no retraining. This replays that pass *in
place*: it chdirs into the existing run directory rather than calling
start_run(), because a `train.no_train=true` re-launch would create a sibling
run instead, leaving the experiment with two runs whose artifacts disagree
(exactly the duplicate-run mess that made wikiann/bert's y-spread wrong before).

Oracle runs are skipped deliberately: their per-word selections are already on
disk as oracle_masks.npz, and re-evaluating one would redo the mask search --
hours per encoder for a file we already have.

  python3 scripts/backfill_selection_log.py [--device cuda] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge.core import ExperimentStore  # noqa: E402

from src.data import initialize_data  # noqa: E402
from src.train import SelectorTrainer  # noqa: E402
from src.utils import (  # noqa: E402
    checkpoint_epoch, configure_runtime, get_logger, start_run_metrics_capture,
)


def pending(store: ExperimentStore):
    for selection in store.all_selections():
        cfg = selection.experiment.config
        if str(OmegaConf.select(cfg, "task")) != "rationale":
            continue
        for run in selection.runs or []:
            if run.status != "done":
                continue
            if (run.path / "data" / "selection_log.npz").exists():
                continue
            ckpts = sorted((run.path / "state/models").glob("model_*.pth"),
                           key=lambda p: checkpoint_epoch(p) or 0)
            if ckpts:
                yield selection.experiment, run, ckpts[-1]


def backfill(experiment, run, checkpoint: Path, device: str) -> None:
    cfg = experiment.config.copy()
    # The run's own runtime snapshot: seed above all, so the eval reproduces
    # the trajectory this checkpoint came from rather than the config default.
    cfg.runtime = OmegaConf.merge(cfg.runtime, run.config)
    cfg.runtime.device = device
    # The run already has spearman_curves.json; this pass exists only to add
    # selection_log.npz, so the sweep would be recomputing what is on disk.
    cfg.runtime.eval.stsb = False
    cfg.runtime, _ = configure_runtime(cfg.runtime)

    cwd = Path.cwd()
    os.chdir(run.path)
    try:
        logger = get_logger("backfill.log")
        train_dl, test_dl, encoder, tokenizer, labels_set, _ = initialize_data(
            cfg.data, cfg.runtime.data, logger, device=cfg.runtime.device,
            keep_special=bool(cfg.model.get("keep_special", True)))
        trainer = SelectorTrainer(cfg, train_dl, test_dl, encoder, tokenizer,
                                  labels_set, logger, run, start_run_metrics_capture())
        trainer.load_checkpoint(checkpoint)
        trainer.final_eval(record_eval_history=False)
    finally:
        os.chdir(cwd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    store = ExperimentStore(root=ROOT / "outputs")
    todo = list(pending(store))
    print(f"{len(todo)} run(s) need data/selection_log.npz", flush=True)
    for experiment, run, ckpt in todo:
        seed = OmegaConf.select(run.config, "seed")
        label = (f"{OmegaConf.select(experiment.config, 'data.dataset')}/"
                 f"{OmegaConf.select(experiment.config, 'data.encoder.family')}/"
                 f"{OmegaConf.select(experiment.config, 'data.encoder.pooling')} seed={seed}")
        print(f"  {run.signature}  {label}"
              + ("   [dry run]" if args.dry_run else ""), flush=True)
        if not args.dry_run:
            backfill(experiment, run, ckpt, args.device)
            ok = (run.path / "data" / "selection_log.npz").exists()
            print(f"      -> selection_log.npz {'written' if ok else 'MISSING'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
