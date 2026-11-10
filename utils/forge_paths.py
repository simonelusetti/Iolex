"""Resolve a forge signature to the run directory the plot scripts read from.

Under dora every experiment owned exactly one directory, so a signature was
a path: `outputs/xps/<sig>/data/foo.json`. Forge separates the two levels --
the *experiment* (the config hash) from the *runs* launched under it
(`outputs/xps/<sig>/<run-id>/`) -- so a signature no longer names a single
directory and the plot scripts have to go through this.

That split is what makes multi-seed sweeps convenient here: `runtime.seed` is
in `forge.exclude` (see conf/config.yaml), so three seeds of one
configuration are three *runs* of one experiment rather than three unrelated
experiments. Pass the experiment signature and `run_dirs` returns all three.

Every `--*-sig` flag accepts either level, and prefixes work the same way
they do in the forge CLI:

    1c3d45ee            the experiment -- all of its completed runs
    1c3d45ee/c8f79e25   one specific run
    1c3d           /    1c3d45ee/c8f      prefixes of either
"""
from __future__ import annotations

import sys
from pathlib import Path

from forge.core import ExperimentStore
from forge.matching import select_signatures

ROOT = Path(__file__).resolve().parent.parent


def _store() -> ExperimentStore:
    # Anchored to the repo root rather than the cwd, so the plot scripts work
    # from anywhere -- forge's own default is cwd-relative.
    return ExperimentStore(root=ROOT / "outputs")


def _known_signatures() -> list[str]:
    return sorted(
        f"{run.signature}  ({run.status})"
        for selection in _store().all_selections()
        for run in (selection.runs or [])
    )


def run_dirs(sig: str) -> list[Path]:
    """Run directories for *sig*, oldest launch first.

    An experiment-level signature resolves to its completed runs; naming a
    run explicitly (`<xp>/<run>`) returns that run whatever its status, since
    asking for it by name is unambiguous about intent.
    """
    store = _store()
    selections = select_signatures([sig], store=store)

    runs = [
        run
        for selection in selections
        for run in (
            selection.runs
            if selection.runs is not None
            else store.list_runs(selection.experiment.signature)
        )
    ]

    named_run = "/" in sig
    if not named_run:
        runs = [run for run in runs if run.status == "done"] or runs

    if not runs:
        known = _known_signatures()
        raise SystemExit(
            f"No run found for signature {sig!r} under {store.xps_dir}.\n"
            + ("Available runs:\n  " + "\n  ".join(known) if known
               else "No runs exist yet -- launch one with `forge run ...`.")
        )

    return [run.path for run in sorted(runs, key=lambda r: r.launched_on)]


def run_dir(sig: str) -> Path:
    """The single run directory for *sig*.

    Scripts that plot one curve per signature use this. If *sig* names an
    experiment with several runs (e.g. a multi-seed sweep), the most recent
    one is used and the choice is reported on stderr rather than made
    silently -- pass `<xp>/<run>` to pin it.
    """
    dirs = run_dirs(sig)
    if len(dirs) > 1:
        chosen = dirs[-1]
        print(
            f"note: signature {sig!r} matches {len(dirs)} runs; using the most "
            f"recent ({chosen.parent.name}/{chosen.name}). "
            f"Pass <xp>/<run> to select a different one.",
            file=sys.stderr,
        )
        return chosen
    return dirs[0]


DEFAULT_TRAIN_RHOS = (0.1, 1.0, 10)


def variant_suffix(cfg) -> str:
    """Label segments for selector variants that would otherwise share a label.

    /prefix-masked -- e5 inputs start with "query: " (src/data.py). Before
        model.selector.keep_nonword existed, the selector masked those prefix
        tokens out of attention at every rho, so the suppressed sentence could
        never match the full one. Those runs are kept but labelled
        "e5/mean/prefix-masked"; the retrained ones (keep_nonword=true) take the
        plain "e5/mean". Only e5 gets a prefix, so every other family is
        unaffected.
    /train-rho<lo>-<hi> -- trained on a budget range other than the default
        model.loss.sweep_range [0.1, 1.0, 10], e.g. "sbert/mean/train-rho0.5-1".
        The range is in the signature, so these are separate experiments; this
        keeps them from sharing a label with the default-range ones.
    /keep-last -- model.selector.keep_last: the last word is protected (always
        kept, outside the budget), e.g. "llm/last/keep-last".
    /next-token -- model.loss.objective=next_token_kl: trained to preserve the
        LM's next-token distribution instead of the pooled representation.
    /tokens -- model.selector.unit=token: subword pieces are selected on
        their own and rho counts pieces, instead of whole words.
    /no-rho -- model.selector.condition_on_rho=false: the scorer does not see
        rho, so one fixed ranking serves every budget (the rho ablation).
    /tau-rank<x>, /tau-gate<x> -- model.selector.tau_rank / tau_gate other than
        the defaults 0.05 / 0.2 (the temperature sensitivity analysis), e.g.
        "sbert/mean/tau-rank0.0125".
    """
    from omegaconf import OmegaConf

    suffix = ""
    if (str(OmegaConf.select(cfg, "data.encoder.family")) == "e5"
            and not OmegaConf.select(cfg, "model.selector.keep_nonword", default=False)):
        suffix += "/prefix-masked"
    rng = OmegaConf.select(cfg, "model.loss.sweep_range", default=None)
    if rng is not None and tuple(float(x) for x in rng) != DEFAULT_TRAIN_RHOS:
        suffix += f"/train-rho{float(rng[0]):g}-{float(rng[1]):g}"
    if OmegaConf.select(cfg, "model.selector.keep_last", default=False):
        suffix += "/keep-last"
    if OmegaConf.select(cfg, "model.loss.objective", default="cosine") == "next_token_kl":
        suffix += "/next-token"
    if OmegaConf.select(cfg, "model.selector.unit", default="word") == "token":
        suffix += "/tokens"
    if not OmegaConf.select(cfg, "model.selector.condition_on_rho", default=True):
        suffix += "/no-rho"
    for key, default in (("tau_rank", 0.05), ("tau_gate", 0.2)):
        tau = float(OmegaConf.select(cfg, f"model.selector.{key}", default=default))
        if tau != default:
            suffix += f"/{key.replace('_', '-')}{tau:g}"
    return suffix


def discover_strategies(dataset: str, tasks: tuple[str, ...] = ("rationale", "oracle")) -> list[dict]:
    """Every selection experiment for *dataset*, newest-config first.

    Shared by the grounding scatter and the bias heatmaps so they can never
    disagree about what exists. Returns one dict per experiment with its
    label, family, pooling, task, signature and completed run directories.

    The label carries a `/<task>` suffix for anything that is not the plain
    trained selector: the brute-force oracle shares a family and pooling with
    the selector it is the ceiling for, so without it the two would collide
    on one key.
    """
    from forge.core import ExperimentStore
    from omegaconf import OmegaConf

    store = ExperimentStore(root=ROOT / "outputs")
    found = []
    for selection in store.all_selections():
        cfg = selection.experiment.config
        if str(OmegaConf.select(cfg, "data.dataset")) != dataset:
            continue
        task = str(OmegaConf.select(cfg, "task"))
        if task not in tasks:
            continue
        done = [r.path for r in (selection.runs or []) if r.status == "done"]
        if not done:
            continue
        family = str(OmegaConf.select(cfg, "data.encoder.family"))
        pooling = OmegaConf.select(cfg, "data.encoder.pooling")
        pooling = None if pooling is None else str(pooling)
        base = (f"{family}/{pooling}" if pooling else family) + variant_suffix(cfg)
        found.append({
            "label": base if task == "rationale" else f"{base}/{task}",
            "family": family,
            "pooling": pooling,
            "task": task,
            "signature": selection.experiment.signature,
            "runs": done,
        })
    return found
