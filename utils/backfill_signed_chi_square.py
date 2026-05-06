"""
backfill_signed_chi_square.py — Generate signed_chi_square_heatmap.json for existing experiments.

Reads chi_square_curves.json and selection_rate_curves.json (already present for every
evaluated experiment) and writes signed_chi_square_heatmap.json to the same data/ dir.
No re-training or re-evaluation required.

Usage:
    python utils/backfill_signed_chi_square.py
    python utils/backfill_signed_chi_square.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import _build_signed_chi_square_heatmap_payload
from utils.dora_utils import XPS_DIR


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill signed_chi_square_heatmap.json for existing experiments.")
    p.add_argument("--dry-run", action="store_true", help="Print what would be written without writing anything.")
    args = p.parse_args()

    sig_dirs = sorted([d for d in XPS_DIR.iterdir() if d.is_dir()])
    saved = skipped_no_source = skipped_empty = already_exist = 0

    for sig_dir in sig_dirs:
        data_dir = sig_dir / "data"
        chi_path = data_dir / "chi_square_curves.json"
        sel_path = data_dir / "selection_rate_curves.json"
        out_path = data_dir / "signed_chi_square_heatmap.json"

        if not chi_path.exists() or not sel_path.exists():
            skipped_no_source += 1
            continue

        if out_path.exists():
            already_exist += 1
            continue

        chi_payload = _load_json(chi_path)
        sel_payload = _load_json(sel_path)

        signed_payload = _build_signed_chi_square_heatmap_payload(chi_payload, sel_payload)

        if not signed_payload.get("curves"):
            print(f"  [skip] {sig_dir.name}: no common labels between artifacts")
            skipped_empty += 1
            continue

        if args.dry_run:
            labels = list(signed_payload["curves"].keys())
            print(f"  [dry-run] {sig_dir.name}: would write {len(labels)} labels × {len(signed_payload['rho'])} rhos")
        else:
            _write_json(out_path, signed_payload)
            print(f"  [saved] {sig_dir.name}")

        saved += 1

    print(f"\nDone: {saved} written, {already_exist} already existed, {skipped_no_source} missing source artifacts, {skipped_empty} empty.")


if __name__ == "__main__":
    main()
