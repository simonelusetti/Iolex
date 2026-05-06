"""
rank_matrix.py — Token-score rank matrix for a trained selector.

Modes:
    python utils/rank_matrix.py --sig 97d170e1
    python utils/rank_matrix.py
    python utils/rank_matrix.py --sig 97d170e1 --plot-only
    python utils/rank_matrix.py --plot-only
    python utils/rank_matrix.py --recompute

Cached data:
    outputs/utils/rank_matrix/data/{sig}/{split}__rank-{N}__samples-{...}__{cls-mode}.json

Plots:
    outputs/utils/rank_matrix/{sig}/{split}.png
    outputs/utils/rank_matrix/overview_{split}.png
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, PowerNorm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import collate, initialize_data
from src.selector import RationaleSelectorModel
from utils.dora_utils import (
    XPS_DIR,
    expected_checkpoint,
    label_from_overrides,
    load_dora_exclude,
    load_overrides_for_sig,
)
from utils.mask_nesting import load_xp_cfg


OUT_ROOT = PROJECT_ROOT / "outputs" / "utils" / "rank_matrix"
DATA_ROOT = OUT_ROOT / "data"

_ENCODER_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}


@dataclass
class RankResult:
    sig: str
    label: str
    split: str
    rhos: list[float]
    z_rank_mean: np.ndarray
    cls_z_mean: np.ndarray
    has_cls: bool
    include_cls_in_rank: bool
    z_rank_n: np.ndarray
    cls_n: np.ndarray


def analyse_sig(sig_dir: Path, args: argparse.Namespace) -> RankResult:
    cfg = load_xp_cfg(sig_dir)

    ckpt_path = expected_checkpoint(sig_dir)
    if ckpt_path is None:
        candidates = sorted((sig_dir / "state" / "models").glob("model_*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found for {sig_dir.name}")
        ckpt_path = candidates[-1]

    keep_special = bool(cfg.model.get("keep_special", True))

    sweep = cfg.model.loss.sweep_range
    rhos = [float(x) for x in np.linspace(sweep[0], sweep[1], int(sweep[2]))]

    R = len(rhos)
    N = args.max_rank

    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    data_cfg.subset = 1.0

    runtime_data_cfg = OmegaConf.create(
        {
            "rebuild": False,
            "test_subset": None,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }
    )

    enc_key = (
        str(cfg.data.encoder.family),
        str(cfg.data.encoder.name),
        keep_special,
        args.device,
    )

    cached_encoder, cached_tokenizer = _ENCODER_CACHE.get(enc_key, (None, None))

    _, _, encoder, tokenizer, _, ds = initialize_data(
        data_cfg,
        runtime_data_cfg,
        device=args.device,
        keep_special=keep_special,
        encoder=cached_encoder,
        tokenizer=cached_tokenizer,
    )

    encoder.eval()
    _ENCODER_CACHE[enc_key] = (encoder, tokenizer)

    dataset = ds[args.split]

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        shuffle=False,
    )

    first_batch = next(iter(loader))

    with torch.no_grad():
        emb_dim = encoder.token_embeddings(
            first_batch["ids"][:1].to(args.device),
            first_batch["attn_mask"][:1].to(args.device),
        ).shape[-1]

    model = RationaleSelectorModel(
        embedding_dim=emb_dim,
        sent_encoder=encoder,
        loss_cfg=OmegaConf.to_container(cfg.model.loss, resolve=True),
        selector_cfg=OmegaConf.to_container(
            cfg.model.get("selector", OmegaConf.create({})),
            resolve=True,
        ),
    ).to(args.device)

    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    model.eval()

    z_rank_sum = np.zeros((R, N), dtype=np.float64)
    z_rank_n = np.zeros(R, dtype=np.int64)

    cls_z_sum = np.zeros(R, dtype=np.float64)
    cls_n = np.zeros(R, dtype=np.int64)

    for batch in tqdm(loader, desc=sig_dir.name[:8], dynamic_ncols=True, leave=False):
        ids = batch["ids"].to(args.device)
        attn = batch["attn_mask"].to(args.device)

        B = ids.shape[0]

        with torch.no_grad():
            token_emb = encoder.token_embeddings(ids, attn)
            z, _, _ = model(ids, token_emb, attn, rhos=rhos)

            if z.ndim == 2 and z.shape[0] == B:
                z = z.unsqueeze(0).expand(R, -1, -1)
            elif z.ndim == 3 and z.shape[:2] == (R, B):
                pass
            elif z.ndim == 3 and z.shape[:2] == (B, R):
                z = z.permute(1, 0, 2)
            else:
                raise ValueError(
                    f"Unexpected z shape {tuple(z.shape)}. "
                    f"Expected [B, L], [R, B, L], or [B, R, L]."
                )

        z_np = z.detach().cpu().float().numpy()
        ids_np = ids.detach().cpu().numpy()
        attn_np = attn.detach().cpu().numpy()

        valid_mask = attn_np.astype(bool)
        valid_len = valid_mask.sum(axis=1)

        cls_id = getattr(tokenizer, "cls_token_id", None)
        cls_mask = (
            ids_np == cls_id
            if cls_id is not None and keep_special
            else np.zeros_like(valid_mask)
        )

        has_cls_example = (cls_mask & valid_mask).any(axis=1)
        cls_positions = cls_mask.argmax(axis=1)

        rank_mask = valid_mask if args.include_cls_in_rank else valid_mask & ~cls_mask

        z_valid = z_np * valid_mask[np.newaxis]
        z_rank = z_np * rank_mask[np.newaxis]

        z_total = z_valid.sum(axis=2)
        total_safe = np.where(z_total > 0, z_total, 1.0)

        z_norm = z_rank / total_safe[:, :, np.newaxis]
        z_sorted = np.sort(z_norm, axis=2)[:, :, ::-1]

        top_n = min(N, z_sorted.shape[2])
        valid_rb = (valid_len > 0)[np.newaxis, :] & (z_total > 0)

        z_rank_sum[:, :top_n] += (
            z_sorted[:, :, :top_n] * valid_rb[:, :, np.newaxis]
        ).sum(axis=1)

        z_rank_n += valid_rb.sum(axis=1)

        if has_cls_example.any():
            z_valid_norm = z_valid / total_safe[:, :, np.newaxis]
            cls_values = z_valid_norm[:, np.arange(B), cls_positions]
            valid_cls = valid_rb & has_cls_example[np.newaxis]

            cls_z_sum += (cls_values * valid_cls).sum(axis=1)
            cls_n += valid_cls.sum(axis=1)

    z_rank_mean = np.full((R, N), np.nan, dtype=np.float64)
    cls_z_mean = np.full(R, np.nan, dtype=np.float64)

    np.divide(
        z_rank_sum,
        z_rank_n[:, None],
        out=z_rank_mean,
        where=z_rank_n[:, None] > 0,
    )

    np.divide(
        cls_z_sum,
        cls_n,
        out=cls_z_mean,
        where=cls_n > 0,
    )

    overrides = load_overrides_for_sig(sig_dir) or []
    label = label_from_overrides(overrides, load_dora_exclude()) or "<default>"

    return RankResult(
        sig=sig_dir.name,
        label=label,
        split=args.split,
        rhos=rhos,
        z_rank_mean=z_rank_mean,
        cls_z_mean=cls_z_mean,
        has_cls=bool(np.any(cls_n > 0)),
        include_cls_in_rank=args.include_cls_in_rank,
        z_rank_n=z_rank_n,
        cls_n=cls_n,
    )


def get_result(sig_dir: Path, args: argparse.Namespace) -> RankResult:
    samples = f"samples-{args.max_samples}" if args.max_samples is not None else "samples-all"
    cls_mode = "cls-in-rank" if args.include_cls_in_rank else "cls-separate"

    cache_file = (
        DATA_ROOT
        / sig_dir.name
        / f"{args.split}__rank-{args.max_rank}__{samples}__{cls_mode}.json"
    )

    if args.plot_only or (cache_file.exists() and not args.recompute):
        with cache_file.open("r", encoding="utf-8") as f:
            p = json.load(f)

        return RankResult(
            sig=p["sig"],
            label=p.get("label", p["sig"]),
            split=p.get("split", args.split),
            rhos=[float(x) for x in p["rhos"]],
            z_rank_mean=np.array(p["z_rank_mean"], dtype=np.float64),
            cls_z_mean=np.array(p["cls_z_mean"], dtype=np.float64),
            has_cls=bool(p["has_cls"]),
            include_cls_in_rank=bool(p.get("include_cls_in_rank", False)),
            z_rank_n=np.array(p["z_rank_n"], dtype=np.int64),
            cls_n=np.array(p["cls_n"], dtype=np.int64),
        )

    result = analyse_sig(sig_dir, args)

    z_rank_json = result.z_rank_mean.astype(object)
    z_rank_json[~np.isfinite(result.z_rank_mean)] = None

    cls_json = result.cls_z_mean.astype(object)
    cls_json[~np.isfinite(result.cls_z_mean)] = None

    cache_file.parent.mkdir(parents=True, exist_ok=True)

    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sig": result.sig,
                "label": result.label,
                "split": result.split,
                "rhos": result.rhos,
                "z_rank_mean": z_rank_json.tolist(),
                "cls_z_mean": cls_json.tolist(),
                "has_cls": result.has_cls,
                "include_cls_in_rank": result.include_cls_in_rank,
                "z_rank_n": result.z_rank_n.tolist(),
                "cls_n": result.cls_n.tolist(),
            },
            f,
            indent=2,
        )

    return result


def plot(results: list[RankResult], args: argparse.Namespace) -> None:
    single = len(results) == 1

    N = results[0].z_rank_mean.shape[1]
    max_R = max(len(result.rhos) for result in results)

    matrices = []

    for result in results:
        matrix = np.full((len(result.rhos), N + 1), np.nan, dtype=np.float64)
        matrix[:, :N] = result.z_rank_mean

        if result.has_cls:
            matrix[:, N] = result.cls_z_mean

        matrices.append(matrix)

    if args.vmax is None:
        values = np.concatenate(
            [m[np.isfinite(m) & (m > 0)] for m in matrices]
        )
        vmax = float(np.percentile(values, args.vmax_percentile))
    else:
        vmax = args.vmax

    norm = (
        Normalize(0, vmax)
        if args.gamma == 1.0
        else PowerNorm(args.gamma, 0, vmax)
    )

    cmap = copy(plt.get_cmap(args.cmap))
    cmap.set_bad("#eeeeee")

    if single:
        nrows, ncols = 1, 1
        figsize = (max(10, (N + 1) * 0.55), max_R * 0.65 + 1.5)
        out_path = args.out or OUT_ROOT / results[0].sig / f"{args.split}.png"
    else:
        ncols = args.ncols
        nrows = -(-len(results) // ncols)
        figsize = (
            ncols * max(3.5, (N + 1) * 0.22) + 1.0,
            nrows * max(2.2, max_R * 0.28) + 0.8,
        )
        out_path = args.out or OUT_ROOT / f"overview_{args.split}.png"

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    im_ref = None

    for idx, (result, matrix) in enumerate(zip(results, matrices)):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, origin="upper")
        im_ref = im if im_ref is None else im_ref

        ax.axvline(N - 0.5, color="white", linewidth=1.5)

        ax.set_xticks(range(N + 1))
        ax.set_xticklabels(
            [str(i + 1) for i in range(N)] + ["CLS"],
            fontsize=5,
            rotation=90,
        )

        ax.set_yticks(range(len(result.rhos)))
        ax.set_yticklabels([f"{rho:.2f}" for rho in result.rhos], fontsize=6)

        if result.has_cls:
            ax.get_xticklabels()[-1].set_color("#1f77b4")

        if single:
            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    value = matrix[r, c]

                    if np.isfinite(value):
                        ax.text(
                            c,
                            r,
                            f"{value:.3f}",
                            ha="center",
                            va="center",
                            fontsize=max(4, 7 - N // 8),
                            color="white" if value > 0.6 * vmax else "black",
                        )

            rank_note = (
                "ranks include CLS"
                if result.include_cls_in_rank
                else "ranks exclude CLS"
            )

            ax.set_xlabel("Token rank  (1 = highest z fraction)                 CLS →", fontsize=9)
            ax.set_ylabel("ρ", fontsize=9)
            ax.set_title(
                f"Rank matrix — {result.sig} [{result.label}]\n"
                f"split={result.split}, {rank_note}",
                fontsize=9,
            )
        else:
            ax.set_title(
                f"{result.sig}\n{result.label}",
                fontsize=5.5,
                loc="left",
                fontfamily="monospace",
            )

    for idx in range(len(results), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    if single:
        fig.colorbar(im_ref, ax=axes[0][0], label="mean normalised z", shrink=0.8)
        fig.tight_layout()
        dpi = 200
    else:
        rank_note = (
            "ranks include CLS"
            if results[0].include_cls_in_rank
            else "ranks exclude CLS"
        )

        fig.suptitle(
            f"Rank-matrix overview — split={args.split}, max_rank={N}, {rank_note}",
            fontsize=10,
        )

        fig.tight_layout(rect=[0, 0, 0.965, 0.965])
        cbar_ax = fig.add_axes([0.975, 0.15, 0.008, 0.7])
        fig.colorbar(im_ref, cax=cbar_ax, label="mean normalised z")
        dpi = 180

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Token-score rank matrix.")

    parser.add_argument("--sig", default=None)
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--ncols", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)

    parser.add_argument("--include-cls-in-rank", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--recompute", action="store_true")

    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--vmax-percentile", type=float, default=99.0)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--gamma", type=float, default=0.7)

    args = parser.parse_args()

    if args.plot_only and args.recompute:
        raise ValueError("--plot-only and --recompute cannot be used together")

    if args.sig is not None:
        results = [get_result(XPS_DIR / args.sig, args)]
    else:
        results = []
        for sig_dir in tqdm(
            sorted(d for d in XPS_DIR.iterdir() if d.is_dir()),
            desc="Signatures",
            dynamic_ncols=True,
        ):
            try:
                results.append(get_result(sig_dir, args))
            except FileNotFoundError:
                tqdm.write(f"  [skip] {sig_dir.name}: no checkpoint")

    plot(results, args)


if __name__ == "__main__":
    main()