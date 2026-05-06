import math
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from matplotlib.colors import LinearSegmentedColormap, FuncNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np


_DEFAULT_LOSS_HISTORY_PATH = Path("data") / "loss_history.json"
_DEFAULT_LOSS_PLOT_PATH = Path("plots") / "loss.png"
_DEFAULT_SELECTION_RATE_CURVES_PATH = Path("data") / "selection_rate_curves.json"
_DEFAULT_CHI_SQUARE_CURVES_PATH = Path("data") / "chi_square_curves.json"
_DEFAULT_CRAMERS_V_CURVES_PATH = Path("data") / "cramers_v_curves.json"
_DEFAULT_SPEARMAN_CURVES_PATH = Path("data") / "spearman_curves.json"
_DEFAULT_SPEARMAN_PLOT_PATH = Path("plots") / "spearman_vs_rho.png"
_DEFAULT_NLI_SPEARMAN_CURVES_PATH = Path("data") / "nli_spearman_curves.json"
_DEFAULT_NLI_SPEARMAN_PLOT_PATH = Path("plots") / "nli_spearman_vs_rho.png"
_DEFAULT_SELECTION_RATE_PLOT_PATH = Path("plots") / "selection_rate_vs_rho.png"
_DEFAULT_CHI_SQUARE_PLOT_PATH = Path("plots") / "chi_square_vs_rho.png"
_DEFAULT_CRAMERS_V_PLOT_PATH = Path("plots") / "cramers_v_vs_rho.png"
_DEFAULT_SIGNED_CHI_SQUARE_HEATMAP_PATH = Path("data") / "signed_chi_square_heatmap.json"
_DEFAULT_SIGNED_CHI_SQUARE_HEATMAP_PLOT_PATH = Path("plots") / "signed_chi_square_heatmap.png"

_METRIC_TO_FILENAME = {
    "selection_rate": "selection_rate_curves.json",
    "chi_square": "chi_square_curves.json",
    "cramers_v": "cramers_v_curves.json",
    "spearman": "spearman_curves.json",
    "nli_spearman": "nli_spearman_curves.json",
    "signed_chi_square": "signed_chi_square_heatmap.json",
}

_METRIC_TO_DATA_PATH = {
    "selection_rate": _DEFAULT_SELECTION_RATE_CURVES_PATH,
    "chi_square": _DEFAULT_CHI_SQUARE_CURVES_PATH,
    "cramers_v": _DEFAULT_CRAMERS_V_CURVES_PATH,
    "spearman": _DEFAULT_SPEARMAN_CURVES_PATH,
    "nli_spearman": _DEFAULT_NLI_SPEARMAN_CURVES_PATH,
    "signed_chi_square": _DEFAULT_SIGNED_CHI_SQUARE_HEATMAP_PATH,
}

_METRIC_TO_PLOT_PATH = {
    "selection_rate": _DEFAULT_SELECTION_RATE_PLOT_PATH,
    "chi_square": _DEFAULT_CHI_SQUARE_PLOT_PATH,
    "cramers_v": _DEFAULT_CRAMERS_V_PLOT_PATH,
    "spearman": _DEFAULT_SPEARMAN_PLOT_PATH,
    "nli_spearman": _DEFAULT_NLI_SPEARMAN_PLOT_PATH,
    "signed_chi_square": _DEFAULT_SIGNED_CHI_SQUARE_HEATMAP_PLOT_PATH,
}


def _legend_in_right_panel(
    ax,
    fontsize: float = 7,
    ncol: int = 1,
    width_ratio: float = 0.72,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * width_ratio, pos.height])
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=fontsize,
        ncol=ncol,
    )

def _plot_metric_from_artifact(metric_name: str, ylabel: str) -> Path:
    data_path = _METRIC_TO_DATA_PATH[metric_name]
    out_path = _METRIC_TO_PLOT_PATH[metric_name]

    metric_payload = _load_json(data_path)
    parsed = maybe_extract_metric_payload(metric_payload)
    if parsed is None:
        raise ValueError(f"Metric missing or invalid in artifact: {data_path}")

    x, curves, baseline = parsed
    if not curves:
        raise ValueError(f"Metric curves are empty in artifact: {data_path}")

    fig, ax = plt.subplots(figsize=(7, 5))

    if isinstance(baseline, Mapping):
        baseline_kind = baseline.get("kind")
        if baseline_kind == "constant":
            try:
                baseline_value = float(baseline.get("value"))
            except (TypeError, ValueError):
                baseline_value = None
            if baseline_value is not None:
                baseline_label = str(baseline.get("label", "baseline"))
                ax.plot(x, [baseline_value] * len(x), "--", label=baseline_label)
        elif baseline_kind == "identity":
            baseline_label = str(baseline.get("label", "baseline"))
            ax.plot(x, x, "--", label=baseline_label)

    display_labels = {
        "selector": "Trained selector",
        "random": "Random selector",
    }
    for label, values in sorted(curves.items(), key=lambda kv: kv[0]):
        ax.plot(x, values, marker="o", linewidth=2.0, label=display_labels.get(label, label))

    ax.set_xlabel("Selection rate (rho)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_selection_rate_curves() -> Path:
    return _plot_metric_from_artifact("selection_rate", "Selection rate")


def plot_chi_square_curves() -> Path:
    return _plot_metric_from_artifact("chi_square", "-log10(p-value)")


def plot_cramers_v_curves() -> Path:
    return _plot_metric_from_artifact("cramers_v", "Cramer's V")


def plot_spearman_curves() -> Path:
    return _plot_metric_from_artifact("spearman", "Spearman correlation (STS-B)")


def plot_nli_spearman_curves() -> Path:
    return _plot_metric_from_artifact("nli_spearman", "Spearman correlation (NLI)")


# ---------------------------------------------------------------------------
# Signed chi-square colormap tuning
# _P05_THRESH  : significance threshold in signed -log10(p) units (~1.301).
# _GREY_FRACTION: fraction of the [0,1] colormap range devoted to the grey
#                 neutral zone (values within ±threshold).  Increase to push
#                 colour further from centre; decrease for a more gradual fade.
# ---------------------------------------------------------------------------
_P05_THRESH    = -math.log10(0.05)   # ≈ 1.301
_GREY_FRACTION = 0.55                # tune this: 0 = no grey zone, 1 = all grey

_SIGNED_CHI_CMAP = LinearSegmentedColormap.from_list(
    "RdGreyBu",
    ["#b2182b", "#d9d9d9", "#2166ac"],
)


def _make_signed_chi_norm(vmax: float) -> FuncNorm:
    """
    Piecewise-linear norm that maps ±_P05_THRESH to the edges of the grey
    centre zone and compresses the rest into the coloured extremes.
    """
    thresh     = _P05_THRESH
    grey_half  = _GREY_FRACTION / 2.0
    color_half = 0.5 - grey_half          # fraction of colormap for each colored side
    safe_v     = max(vmax - thresh, 1e-8)
    safe_t     = max(thresh, 1e-8)

    def forward(x):
        x   = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        pm  = x >= thresh
        nm  = x <= -thresh
        dm  = ~pm & ~nm
        out[pm] = (0.5 + grey_half) + color_half * (x[pm] - thresh) / safe_v
        out[nm] = color_half * (x[nm] + vmax) / safe_v
        out[dm] = 0.5 + grey_half * x[dm] / safe_t
        return np.clip(out, 0.0, 1.0)

    def inverse(y):
        y   = np.asarray(y, dtype=float)
        out = np.empty_like(y)
        pm  = y >= 0.5 + grey_half
        nm  = y <= 0.5 - grey_half
        dm  = ~pm & ~nm
        out[pm] = thresh + safe_v * (y[pm] - (0.5 + grey_half)) / max(color_half, 1e-8)
        out[nm] = -vmax  + safe_v * y[nm] / max(color_half, 1e-8)
        out[dm] = safe_t * (y[dm] - 0.5) / max(grey_half, 1e-8)
        return out

    return FuncNorm((forward, inverse), vmin=-vmax, vmax=vmax)


def _signed_chi_square_heatmap_from_payload(
    payload: Mapping[str, Any],
    ax: plt.Axes,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: FuncNorm | None = None,
) -> plt.cm.ScalarMappable:
    """Render signed chi-square heatmap onto ax. Returns the image for colorbar reuse."""
    rho_values = [float(r) for r in payload.get("rho", [])]
    curves_raw = payload.get("curves", {})

    labels = sorted(curves_raw.keys())
    if not labels or not rho_values:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return plt.cm.ScalarMappable(cmap=_SIGNED_CHI_CMAP)

    matrix = np.array([[float(v) for v in curves_raw[lbl]] for lbl in labels])  # [L, R]

    if vmax is None:
        vmax = float(np.nanmax(np.abs(matrix))) or 1.0
    if norm is None:
        norm = _make_signed_chi_norm(vmax)

    im = ax.imshow(matrix, aspect="auto", cmap=_SIGNED_CHI_CMAP, norm=norm, origin="upper")
    ax.set_xticks(range(len(rho_values)))
    ax.set_xticklabels([f"{r:.2f}" for r in rho_values], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    return im


def plot_signed_chi_square_heatmap() -> Path:
    out_path = _DEFAULT_SIGNED_CHI_SQUARE_HEATMAP_PLOT_PATH
    payload = _load_json(_DEFAULT_SIGNED_CHI_SQUARE_HEATMAP_PATH)

    rho_values = payload.get("rho", [])
    curves_raw = payload.get("curves", {})
    if not rho_values or not curves_raw:
        raise ValueError(f"Metric missing or invalid in artifact: {_DEFAULT_SIGNED_CHI_SQUARE_HEATMAP_PATH}")

    n_labels = len(curves_raw)
    fig_h = max(3.0, n_labels * 0.4 + 1.5)
    fig_w = max(6.0, len(rho_values) * 0.55 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = _signed_chi_square_heatmap_from_payload(payload, ax)
    ax.set_xlabel("Selection rate (ρ)", fontsize=9)
    ax.set_title("Signed chi-square  (blue = over-selected, red = under-selected)", fontsize=9)
    fig.colorbar(im, ax=ax, label="sign × −log₁₀(p)", shrink=0.8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_eval_plots(metric_names: Sequence[str]) -> dict[str, Path]:
    _plotters = {
        "selection_rate": plot_selection_rate_curves,
        "chi_square": plot_chi_square_curves,
        "cramers_v": plot_cramers_v_curves,
        "spearman": plot_spearman_curves,
        "nli_spearman": plot_nli_spearman_curves,
        "signed_chi_square": plot_signed_chi_square_heatmap,
    }
    plot_paths: dict[str, Path] = {}
    for metric_name in metric_names:
        plotter = _plotters.get(metric_name)
        if plotter is None:
            continue
        try:
            plot_paths[metric_name] = plotter()
        except ValueError:
            pass  # artifact exists but has no data (e.g. unlabelled dataset)
    return plot_paths


def save_train_eval_loss_plot(
    ema_alpha: float = 0.2,
) -> None:
    payload = _load_json(_DEFAULT_LOSS_HISTORY_PATH)
    train_loss_history = payload.get("train", []) if isinstance(payload, Mapping) else []
    eval_loss_history = payload.get("eval", []) if isinstance(payload, Mapping) else []

    if not train_loss_history and not eval_loss_history:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    train_ax, eval_ax = axes

    alpha = float(ema_alpha)
    if not (0.0 < alpha <= 1.0):
        alpha = 0.2

    def _plot_history(ax, history: Sequence[Mapping[str, float]], title: str) -> None:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        if not history:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        epochs = range(1, len(history) + 1)
        loss_keys = list(history[0].keys())
        for key in loss_keys:
            ys = [float(entry.get(key, np.nan)) for entry in history]
            ax.plot(epochs, ys, alpha=0.45, linewidth=1.5, label=f"{key} (raw)")
            ys_ema = _ema(ys, alpha)
            ax.plot(epochs, ys_ema, linewidth=2.2, label=f"{key} (EMA {alpha:.2f})")

        values = [float(v) for entry in history for v in entry.values()]
        if values:
            vmin = min(values)
            vmax = max(values)
            if vmax <= vmin:
                vmax = vmin + 1.0
            ax.set_ylim(vmin, vmax * 1.05)
        ax.legend(fontsize="small")

    _plot_history(train_ax, train_loss_history, "Train Losses")
    _plot_history(eval_ax, eval_loss_history, "Eval Losses")

    fig.tight_layout()
    fig.savefig(_DEFAULT_LOSS_PLOT_PATH, bbox_inches="tight")
    plt.close(fig)


def mean_std_curves(curves: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    if not curves:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    max_len = max(len(c) for c in curves)
    arr = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, curve in enumerate(curves):
        arr[i, : len(curve)] = np.asarray(curve, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, std


def _ema(values: Sequence[float], alpha: float) -> list[float]:
    if not values:
        return []
    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(alpha * float(value) + (1.0 - alpha) * smoothed[-1])
    return smoothed


def plot_with_band(
    ax,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
    linestyle: str = "-",
    alpha: float = 0.18,
) -> None:
    valid = np.isfinite(mean)
    if not np.any(valid):
        return
    xv = x[valid]
    yv = mean[valid]
    sv = std[valid]
    line, = ax.plot(xv, yv, marker="o", linewidth=2.0, linestyle=linestyle, label=label)
    band_valid = np.isfinite(sv)
    if np.any(band_valid):
        xb = xv[band_valid]
        yb = yv[band_valid]
        sb = sv[band_valid]
        ax.fill_between(xb, yb - sb, yb + sb, alpha=alpha, color=line.get_color())


def _build_overview_figure(n_groups: int, ncols: int, width: float = 5.8, height: float = 4.6):
    nrows = max(1, math.ceil(n_groups / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * width, nrows * height))
    fig.subplots_adjust(wspace=35, hspace=5)
    return fig, np.asarray(axes).reshape(-1)


def _setup_overview_axis(ax, label: str, n_runs: int, xlabel: str, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_title(f"{label}\\nn={n_runs}", fontsize=8, loc="left", fontfamily="monospace")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _finalize_overview_figure(fig, axes: np.ndarray, n_groups: int, out_path: Path, dpi: int = 180) -> None:
    for ax in axes[n_groups:]:
        ax.set_visible(False)
    fig.tight_layout(pad=1.1, w_pad=2.2, h_pad=2.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_group_label_curves(ax, x_ref: np.ndarray, per_label_runs: dict[str, list[np.ndarray]]) -> None:
    for label, curves in sorted(per_label_runs.items(), key=lambda kv: kv[0]):
        mean, std = mean_std_curves([c.tolist() for c in curves])
        plot_with_band(ax, x_ref, mean, std, f"{label} (n={len(curves)})")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_loss_histories_for_run(run_dir: Path) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    loss_path = run_dir / "data" / "loss_history.json"
    if not loss_path.exists():
        return [], []
    payload = _load_json(loss_path)
    train_history = payload.get("train", []) if isinstance(payload, Mapping) else []
    eval_history = payload.get("eval", []) if isinstance(payload, Mapping) else []
    if not isinstance(train_history, list) or not isinstance(eval_history, list):
        return [], []
    return train_history, eval_history


def _load_metric_payload_for_run(run_dir: Path, metric: str) -> Mapping[str, Any] | None:
    filename = _METRIC_TO_FILENAME.get(metric)
    if filename is None:
        return None
    payload_path = run_dir / "data" / filename
    if not payload_path.exists():
        return None
    payload = _load_json(payload_path)
    return payload if isinstance(payload, Mapping) else None


def extract_metric_payload(metric_payload: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any] | None]:
    rho_values = metric_payload.get("rho")
    curves_raw = metric_payload.get("curves")
    baseline = metric_payload.get("baseline")

    if not isinstance(rho_values, list) or not rho_values:
        raise ValueError("Metric payload has invalid rho grid")
    if not isinstance(curves_raw, Mapping) or not curves_raw:
        raise ValueError("Metric payload has invalid curves mapping")

    x = np.asarray([float(v) for v in rho_values], dtype=float)
    curves = {str(label): np.asarray([float(v) for v in values], dtype=float) for label, values in curves_raw.items()}
    return x, curves, baseline if isinstance(baseline, Mapping) else None


def maybe_extract_metric_payload(
    metric_payload: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any] | None] | None:
    try:
        return extract_metric_payload(metric_payload)
    except ValueError:
        return None


def plot_loss_overview(groups: Sequence[Any], out_path: Path, ncols: int) -> None:
    def _plot_loss_ax(
        ax,
        histories: Sequence[Sequence[Mapping[str, float]]],
        metric_key: str,
        title: str | None = None,
        xlabel: bool = False,
    ) -> None:
        curves = [[entry[metric_key] for entry in h if metric_key in entry] for h in histories]
        curves = [c for c in curves if c]

        ax.grid(True, alpha=0.2)
        ax.set_ylabel(metric_key.replace("_", " "), fontsize=7)
        if title:
            ax.set_title(f"{title}\\nn={len(curves)}", fontsize=8, loc="left", fontfamily="monospace")
        if xlabel:
            ax.set_xlabel("epoch", fontsize=7)

        if not curves:
            ax.text(0.5, 0.5, f"no {metric_key}", transform=ax.transAxes, ha="center", va="center", fontsize=7)
            return

        mean, std = mean_std_curves(curves)
        x = np.arange(1, len(mean) + 1, dtype=float)
        plot_with_band(ax, x, mean, std, "mean±std")
        ema_mean = _ema(mean.tolist(), 0.2)
        ax.plot(x[: len(ema_mean)], ema_mean, linewidth=2.2, label="EMA 0.20")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=6)

    n = len(groups)
    nrows = max(1, math.ceil(n / ncols))
    fig = plt.figure(figsize=(ncols * 6.0, nrows * 8.8))
    outer_gs = fig.add_gridspec(nrows, ncols, hspace=0.7, wspace=0.5)

    for i, group in enumerate(groups):
        row, col = divmod(i, ncols)
        inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[row, col], hspace=0.45)
        train_ax = fig.add_subplot(inner_gs[0])
        eval_ax = fig.add_subplot(inner_gs[1])
        loaded_histories = [_load_loss_histories_for_run(run.sig_dir) for run in group.runs]
        _plot_loss_ax(train_ax, [train_h for train_h, _ in loaded_histories], "train_loss", title=group.label)
        _plot_loss_ax(eval_ax, [eval_h for _, eval_h in loaded_histories], "eval_loss", xlabel=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_spearman_overview(groups: Sequence[Any], out_path: Path, ncols: int) -> None:
    fig, axes = _build_overview_figure(len(groups), ncols)

    for ax, group in zip(axes, groups):
        _setup_overview_axis(ax, group.label, len(group.runs), "selection rate (rho)", "spearman")

        selector_curves: list[np.ndarray] = []
        random_curves: list[np.ndarray] = []
        x_ref: np.ndarray | None = None
        skipped_mismatch = 0

        for run in group.runs:
            metric_payload = _load_metric_payload_for_run(run.sig_dir, metric="spearman")
            if metric_payload is None:
                continue
            parsed = maybe_extract_metric_payload(metric_payload)
            if parsed is None:
                continue
            x, curves, _ = parsed
            y_selector = curves.get("selector")
            y_random = curves.get("random")
            if y_selector is None or y_random is None:
                continue
            if x_ref is None:
                x_ref = x
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                skipped_mismatch += 1
                continue
            selector_curves.append(y_selector)
            random_curves.append(y_random)

        if skipped_mismatch:
            print(f"Skipped {skipped_mismatch} spearman runs in group '{group.label}' due to rho-grid mismatch")

        if not selector_curves or x_ref is None:
            ax.text(0.5, 0.5, "no spearman data", transform=ax.transAxes, ha="center", va="center")
            continue

        selector_mean, selector_std = mean_std_curves([c.tolist() for c in selector_curves])
        random_mean, random_std = mean_std_curves([c.tolist() for c in random_curves])
        plot_with_band(ax, x_ref, selector_mean, selector_std, "selector mean+-std")
        plot_with_band(ax, x_ref, random_mean, random_std, "random mean+-std", linestyle="--", alpha=0.14)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)

    _finalize_overview_figure(fig, axes, len(groups), out_path)


def plot_nli_spearman_overview(groups: Sequence[Any], out_path: Path, ncols: int) -> None:
    fig, axes = _build_overview_figure(len(groups), ncols)

    for ax, group in zip(axes, groups):
        _setup_overview_axis(ax, group.label, len(group.runs), "selection rate (rho)", "spearman (NLI)")

        selector_curves: list[np.ndarray] = []
        random_curves: list[np.ndarray] = []
        x_ref: np.ndarray | None = None
        skipped_mismatch = 0

        for run in group.runs:
            metric_payload = _load_metric_payload_for_run(run.sig_dir, metric="nli_spearman")
            if metric_payload is None:
                continue
            parsed = maybe_extract_metric_payload(metric_payload)
            if parsed is None:
                continue
            x, curves, _ = parsed
            y_selector = curves.get("selector")
            y_random = curves.get("random")
            if y_selector is None or y_random is None:
                continue
            if x_ref is None:
                x_ref = x
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                skipped_mismatch += 1
                continue
            selector_curves.append(y_selector)
            random_curves.append(y_random)

        if skipped_mismatch:
            print(f"Skipped {skipped_mismatch} nli_spearman runs in group '{group.label}' due to rho-grid mismatch")

        if not selector_curves or x_ref is None:
            ax.text(0.5, 0.5, "no nli_spearman data", transform=ax.transAxes, ha="center", va="center")
            continue

        selector_mean, selector_std = mean_std_curves([c.tolist() for c in selector_curves])
        random_mean, random_std = mean_std_curves([c.tolist() for c in random_curves])
        plot_with_band(ax, x_ref, selector_mean, selector_std, "selector mean+-std")
        plot_with_band(ax, x_ref, random_mean, random_std, "random mean+-std", linestyle="--", alpha=0.14)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)

    _finalize_overview_figure(fig, axes, len(groups), out_path)


def _load_signed_chi_square_payload_for_run(run_dir: Path) -> Mapping[str, Any] | None:
    """Load signed_chi_square_heatmap.json, falling back to computing it on-the-fly."""
    from src.eval import _build_signed_chi_square_heatmap_payload

    payload = _load_metric_payload_for_run(run_dir, metric="signed_chi_square")
    if payload is not None and payload.get("curves"):
        return payload

    # Fallback: compute from the two source artifacts that are always present
    chi_payload = _load_metric_payload_for_run(run_dir, metric="chi_square")
    sel_payload = _load_metric_payload_for_run(run_dir, metric="selection_rate")
    if chi_payload is None or sel_payload is None:
        return None

    computed = _build_signed_chi_square_heatmap_payload(chi_payload, sel_payload)
    return computed if computed.get("curves") else None


def plot_signed_chi_square_heatmap_overview(groups: Sequence[Any], out_path: Path, ncols: int) -> None:
    all_abs_vals: list[float] = []
    group_matrices: list[tuple[Any, list[float], dict[str, list[float]]]] = []

    for group in groups:
        label_accum: dict[str, list[list[float]]] = {}
        rho_ref: list[float] | None = None

        for run in group.runs:
            payload = _load_signed_chi_square_payload_for_run(run.sig_dir)
            if payload is None:
                continue
            parsed = maybe_extract_metric_payload(payload)
            if parsed is None:
                continue
            x, curves, _ = parsed
            if rho_ref is None:
                rho_ref = x.tolist()
            for label, curve in curves.items():
                label_accum.setdefault(label, []).append(curve.tolist())
                all_abs_vals.extend(float(abs(v)) for v in curve.tolist() if np.isfinite(v))

        if rho_ref is None:
            group_matrices.append((group, [], {}))
            continue

        mean_curves: dict[str, list[float]] = {
            lbl: list(np.mean(np.array(runs_data), axis=0))
            for lbl, runs_data in label_accum.items()
        }
        group_matrices.append((group, rho_ref, mean_curves))

    global_vmax = float(np.percentile(all_abs_vals, 95)) if all_abs_vals else 1.0
    shared_norm = _make_signed_chi_norm(global_vmax)

    fig, axes = _build_overview_figure(len(groups), ncols, width=5.5, height=4.2)

    im_ref = None
    for ax, (group, rho_ref, mean_curves) in zip(axes, group_matrices):
        ax.set_title(
            f"{group.label}\\nn={len(group.runs)}",
            fontsize=8, loc="left", fontfamily="monospace",
        )
        if not rho_ref or not mean_curves:
            ax.text(0.5, 0.5, "no signed_chi_square data", ha="center", va="center", transform=ax.transAxes)
            continue

        pseudo_payload = {"rho": rho_ref, "curves": mean_curves}
        im = _signed_chi_square_heatmap_from_payload(
            pseudo_payload, ax, vmax=global_vmax, norm=shared_norm
        )
        ax.set_xlabel("ρ", fontsize=7)
        if im_ref is None:
            im_ref = im

    # Hide unused axes
    for ax in axes[len(groups):]:
        ax.set_visible(False)

    # Layout: reserve right margin for colorbar
    fig.tight_layout(pad=1.1, w_pad=2.2, h_pad=2.2, rect=[0, 0, 0.92, 1])

    # Colorbar in dedicated right-margin axis
    cbar_ax = fig.add_axes([0.935, 0.15, 0.012, 0.7])
    if im_ref is not None:
        cbar = fig.colorbar(im_ref, cax=cbar_ax)
    else:
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-global_vmax, vmax=global_vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)

    cbar.set_label("sign × −log₁₀(p)", fontsize=8)

    # Mark ±p=0.05 threshold on the colorbar
    p_thresh = -math.log10(0.05)  # ≈ 1.301
    for y in (p_thresh, -p_thresh):
        cbar_ax.axhline(y=y, color="black", linestyle="--", linewidth=1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_chi_square_overview(groups: Sequence[Any], out_path: Path, ncols: int, metric: str) -> None:
    ylabel = "-log10(p)" if metric == "chi_square" else "Cramer's V"
    fig, axes = _build_overview_figure(len(groups), ncols, width=7.2)

    for ax, group in zip(axes, groups):
        _setup_overview_axis(ax, group.label, len(group.runs), "selection rate", ylabel)
        x_ref: np.ndarray | None = None
        per_label_runs: dict[str, list[np.ndarray]] = {}
        baselines: list[float] = []

        for run in group.runs:
            metric_payload = _load_metric_payload_for_run(run.sig_dir, metric=metric)
            if metric_payload is None:
                continue
            parsed = maybe_extract_metric_payload(metric_payload)
            if parsed is None:
                continue
            x, label_curves, baseline = parsed
            if isinstance(baseline, Mapping) and baseline.get("kind") == "constant":
                try:
                    baselines.append(float(baseline.get("value")))
                except (TypeError, ValueError):
                    pass
            if x_ref is None:
                x_ref = x
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                raise ValueError(f"Chi-square rho grid mismatch inside group: {group.label}")
            for label, curve in label_curves.items():
                per_label_runs.setdefault(label, []).append(curve)

        if x_ref is None or not per_label_runs:
            ax.text(0.5, 0.5, "no chi-square data", transform=ax.transAxes, ha="center", va="center")
            continue

        _plot_group_label_curves(ax, x_ref, per_label_runs)

        if baselines:
            ax.axhline(float(np.mean(baselines)), linestyle="--", linewidth=1.5, color="0.35", label="p=0.05")
        _legend_in_right_panel(ax, fontsize=6)

    _finalize_overview_figure(fig, axes, len(groups), out_path)


def plot_selection_rates_overview(groups: Sequence[Any], out_path: Path, ncols: int) -> None:
    fig, axes = _build_overview_figure(len(groups), ncols, width=7.2)

    for ax, group in zip(axes, groups):
        _setup_overview_axis(ax, group.label, len(group.runs), "effective selection rate (rho)", "selection rate", ylim=(0.0, 1.05))
        x_ref: np.ndarray | None = None
        per_label_runs: dict[str, list[np.ndarray]] = {}
        show_identity_baseline = False

        for run in group.runs:
            metric_payload = _load_metric_payload_for_run(run.sig_dir, metric="selection_rate")
            if metric_payload is None:
                continue
            parsed = maybe_extract_metric_payload(metric_payload)
            if parsed is None:
                continue
            x, label_curves, baseline = parsed
            if isinstance(baseline, Mapping) and baseline.get("kind") == "identity":
                show_identity_baseline = True
            if x_ref is None:
                x_ref = x
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                raise ValueError(f"Selection-rate rho grid mismatch inside group: {group.label}")
            for label, curve in label_curves.items():
                per_label_runs.setdefault(label, []).append(curve)

        if x_ref is None or not per_label_runs:
            ax.text(0.5, 0.5, "no selections data", transform=ax.transAxes, ha="center", va="center")
            continue

        if show_identity_baseline:
            ax.plot(x_ref, x_ref, linestyle="--", linewidth=1.5, color="0.35", label="baseline (y=x)")
        _plot_group_label_curves(ax, x_ref, per_label_runs)
        _legend_in_right_panel(ax, fontsize=6)

    _finalize_overview_figure(fig, axes, len(groups), out_path)


