from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.view import (
    _load_metric_payload_for_run,
    maybe_extract_metric_payload,
    plot_chi_square_overview,
    plot_loss_overview,
    plot_nli_spearman_overview,
    plot_selection_rates_overview,
    plot_signed_chi_square_heatmap_overview,
    plot_spearman_combined,
    plot_spearman_overview,
)
from utils.dora_utils import (
    XPS_DIR,
    build_groups,
    expected_checkpoint,
    filter_sig_dirs_by_group_size,
    load_dora_exclude,
    load_overrides_for_sig,
    load_run,
    needs_eval,
    rerun_eval,
)


OUT_ROOT = PROJECT_ROOT / "outputs" / "utils" / "overview"


def _filter_groups_for_metric(groups, metric: str) -> list:
    """Return only groups where at least one run has non-empty curves for this metric."""
    result = []
    for group in groups:
        for run in group.runs:
            payload = _load_metric_payload_for_run(run.sig_dir, metric=metric)
            if payload is None:
                continue
            parsed = maybe_extract_metric_payload(payload)
            if parsed is not None:
                _, curves, _ = parsed
                if curves:
                    result.append(group)
                    break
    return result


def _get_data_subset(overrides: list[str]) -> float:
    for item in reversed(overrides):
        if item.startswith("data.subset="):
            try:
                return float(item.split("=", 1)[1])
            except ValueError:
                pass
    return 1.0  # not overridden → config default


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render grouped overview figures (loss, chi-square, spearman) as mean+-std across runs."
    )
    parser.add_argument("--sigs", nargs="*", default=None, help="Optional list of signatures to include.")
    parser.add_argument("--rerun-eval", action="store_true", help="Force re-run eval on each selected signature.")
    parser.add_argument("--ncols", type=int, default=4, help="Grid columns for overview figures.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Custom output directory.")
    parser.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="Keep only groups with at least this many runs (after Dora excludes and run key filtering).",
    )
    parser.add_argument(
        "--only-multi-run-groups",
        action="store_true",
        help="Convenience flag for --min-runs=2.",
    )
    parser.add_argument(
        "--all-subsets",
        dest="full_dataset_only",
        action="store_false",
        help="Include experiments with data.subset != 1.0 (excluded by default).",
    )
    parser.add_argument(
        "--titles",
        nargs="+",
        default=None,
        metavar="TITLE",
        help="Custom titles for each panel, in group order. Panels without a matching title keep the auto-generated label.",
    )
    parser.add_argument(
        "--combined-spearman",
        action="store_true",
        help="Produce a single combined Spearman plot with one curve per group instead of the regular overview figures.",
    )
    parser.add_argument(
        "--group-labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="Labels for each group curve in the combined Spearman plot, in group order.",
    )
    parser.add_argument(
        "--single-random",
        action="store_true",
        help="In --combined-spearman mode, pool all groups' random curves into one averaged grey curve.",
    )
    parser.set_defaults(full_dataset_only=True)
    args = parser.parse_args()

    if args.min_runs < 1:
        raise ValueError("--min-runs must be >= 1")

    min_group_runs = max(args.min_runs, 2) if args.only_multi_run_groups else args.min_runs

    if args.sigs:
        sig_dirs = [XPS_DIR / sig for sig in args.sigs]
    else:
        sig_dirs = sorted([p for p in XPS_DIR.iterdir() if p.is_dir()], key=lambda p: p.name)

    for sig_dir in sig_dirs:
        if not sig_dir.exists():
            raise FileNotFoundError(f"Missing signature directory: {sig_dir}")

    if args.full_dataset_only:
        before = len(sig_dirs)
        sig_dirs = [
            d for d in sig_dirs
            if _get_data_subset(load_overrides_for_sig(d) or []) == 1.0
        ]
        excluded = before - len(sig_dirs)
        if excluded:
            print(f"Excluded {excluded} signature(s) with data.subset != 1.0 (pass --all-subsets to include).")

    exclude_patterns = load_dora_exclude()
    sig_dirs = filter_sig_dirs_by_group_size(sig_dirs, exclude_patterns, min_group_runs)
    if not sig_dirs:
        raise ValueError("No signatures left after group-size filtering.")

    if args.rerun_eval:
        print(f"Force re-running evaluation for all {len(sig_dirs)} selected signatures...")
        for sig_dir in sig_dirs:
            rerun_eval(sig_dir.name, sig_dir, expected_checkpoint(sig_dir), dry_run=False)
        print()
    else:
        missing_eval = [sd for sd in sig_dirs if needs_eval(sd)]
        if missing_eval:
            print(f"Found {len(missing_eval)} runs with missing evaluation artifacts.")
            print("Auto-triggering evaluation...")
            for sig_dir in missing_eval:
                rerun_eval(sig_dir.name, sig_dir, expected_checkpoint(sig_dir), dry_run=False)
            print()

    runs = []
    skipped_missing_eval = 0
    for sig_dir in sig_dirs:
        if needs_eval(sig_dir):
            skipped_missing_eval += 1
            continue
        try:
            runs.append(load_run(sig_dir))
        except FileNotFoundError as exc:
            print(f"Skipping {sig_dir.name}: {exc}")

    if skipped_missing_eval:
        print(f"Skipped {skipped_missing_eval} runs still missing evaluation artifacts.")

    if not runs:
        raise ValueError("No valid runs found after loading signatures.")

    groups = build_groups(runs, exclude_patterns)
    if not groups:
        raise ValueError("No experiment groups found.")

    out_root = args.output_dir or OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    for child in out_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)

    if args.combined_spearman:
        plot_spearman_combined(
            groups,
            out_root / "spearman_combined.png",
            group_labels=args.group_labels,
            single_random=args.single_random,
        )
        print(f"Saved combined Spearman plot to {out_root / 'spearman_combined.png'}")
        return

    titles = args.titles

    plot_loss_overview(groups, out_root / "loss_overview.png", ncols=args.ncols, titles=titles)
    plot_spearman_overview(groups, out_root / "spearman_overview.png", ncols=args.ncols, titles=titles)
    plot_nli_spearman_overview(groups, out_root / "nli_spearman_overview.png", ncols=args.ncols, titles=titles)

    sel_groups = _filter_groups_for_metric(groups, "selection_rate")
    if sel_groups:
        plot_selection_rates_overview(sel_groups, out_root / "selection_rates_overview.png", ncols=args.ncols, titles=titles)
    else:
        print("Skipping selection_rate overview (no labeled datasets).")

    chi_groups = _filter_groups_for_metric(groups, "chi_square")
    if chi_groups:
        plot_chi_square_overview(chi_groups, out_root / "chi_square_overview.png", ncols=args.ncols, metric="chi_square", titles=titles)
        plot_chi_square_overview(chi_groups, out_root / "cramers_v_overview.png", ncols=args.ncols, metric="cramers_v", titles=titles)
        plot_signed_chi_square_heatmap_overview(chi_groups, out_root / "signed_chi_square_heatmap_overview.png", ncols=args.ncols, titles=titles)
    else:
        print("Skipping chi_square / cramers_v / signed_chi_square overviews (no labeled datasets).")

    print(f"Saved overview figures to {out_root}")


if __name__ == "__main__":
    main()
