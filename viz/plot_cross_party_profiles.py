"""Cross-party similarity profiles at aligned α.

When steering as party X, each figure shows how similarity to *all five*
judged parties moves. α is held fixed in snapshot plots and is the x-axis
in the dose-response lines. At a given α the layer is the one with the
largest target-party Δ among the three probe-selected layers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_utils import (
    ALPHA_GRID,
    PARTIES,
    PARTY_ORDER,
    available_models,
    load_cross_party_fixed_alpha_table,
    model_label,
    output_dir,
)

sns.set_theme(style="whitegrid", context="paper")

SNAPSHOT_ALPHAS = tuple(a for a in ALPHA_GRID if a > 0)
LINE_ALPHAS = ALPHA_GRID
PARTY_COLORS = list(plt.cm.tab10.colors[: len(PARTY_ORDER)])


def _out() -> Path:
    path = output_dir() / "cross_party_profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_out(model: str) -> Path:
    path = _out() / model
    path.mkdir(parents=True, exist_ok=True)
    return path


def _alpha_tag(alpha: float) -> str:
    return "alpha_" + f"{alpha:g}".replace(".", "p")


def _alpha_title(alpha: float) -> str:
    return f"α = {alpha:g}"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _grid(n: int, cell: Tuple[float, float] = (4.6, 3.8), max_cols: int = 5):
    ncols = min(max_cols, max(n, 1))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows), squeeze=False)
    return fig, axes.flatten()


def _hide_extra(axes: Iterable[plt.Axes], used: int) -> None:
    for ax in list(axes)[used:]:
        ax.set_visible(False)


def _slice_alpha(table: pd.DataFrame, alpha: float) -> pd.DataFrame:
    return table[np.isclose(table["alpha"], alpha)].copy()


def _shared_ylim(values: np.ndarray, pad: float = 0.12) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-0.5, 1.0)
    lo = min(0.0, float(np.min(finite)))
    hi = max(0.0, float(np.max(finite)))
    span = max(hi - lo, 0.4)
    return lo - pad * span, hi + pad * span


def _judged_vector(frame: pd.DataFrame, steered: str, value: str) -> pd.Series:
    sub = frame[frame["steered_party_key"] == steered]
    return sub.set_index("compared_party_key").reindex(PARTY_ORDER)[value]


def plot_one_steered_party(
    model: str,
    steered: str,
    alpha: float,
    frame: pd.DataFrame,
    delta_ylim: Tuple[float, float],
) -> None:
    """One figure: steering as ``steered`` at a fixed α, effect on all judged parties."""
    sub = frame[
        (frame["model"] == model)
        & (frame["steered_party_key"] == steered)
        & np.isclose(frame["alpha"], alpha)
    ]
    if sub.empty:
        return
    sub = sub.set_index("compared_party_key").reindex(PARTY_ORDER)
    labels = [
        PARTIES[pk] + (" (target)" if pk == steered else "")
        for pk in PARTY_ORDER
    ]
    layer = int(sub["layer"].iloc[0])
    n_pairs = int(sub["n_pairs"].iloc[0]) if pd.notna(sub["n_pairs"].iloc[0]) else None
    incomplete = bool(sub["incomplete_pairs"].iloc[0])

    fig, (ax_score, ax_delta) = plt.subplots(2, 1, figsize=(8.6, 8.2), sharex=True)
    fig.suptitle(
        f"{model_label(model)} — steering as {PARTIES[steered]}\n"
        f"{_alpha_title(alpha)}  ·  layer {layer}"
        + (f"  ·  n={n_pairs} pairs" if incomplete else ""),
        fontweight="bold",
        fontsize=13,
    )

    x = np.arange(len(PARTY_ORDER))
    width = 0.36
    alpha0 = sub["alpha0_score"].to_numpy(dtype=float)
    steered_scores = sub["mean_score"].to_numpy(dtype=float)
    deltas = sub["mean_delta"].to_numpy(dtype=float)

    ax_score.bar(x - width / 2, alpha0, width, label="Unsteered (α=0)", color="#C44E52", edgecolor="white")
    ax_score.bar(x + width / 2, steered_scores, width, label="Steered", color="#4C72B0", edgecolor="white")
    ax_score.set_ylim(0, 10)
    ax_score.set_ylabel("Mean 1–10 similarity")
    ax_score.legend(loc="upper right", fontsize=8)
    ax_score.set_title("Similarity to each party", loc="left", fontsize=11)
    for i, (base, steered_val, delta) in enumerate(zip(alpha0, steered_scores, deltas)):
        if pd.notna(delta):
            ax_score.annotate(
                f"Δ={delta:+.2f}",
                xy=(i + width / 2, steered_val if pd.notna(steered_val) else 0),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    bar_colors = ["#2A9D8F" if pk == steered else "#4C72B0" for pk in PARTY_ORDER]
    ax_delta.bar(x, deltas, color=bar_colors, edgecolor="white")
    ax_delta.axhline(0, color="black", lw=0.8)
    ax_delta.set_ylim(*delta_ylim)
    ax_delta.set_ylabel("Mean Δ vs reused α=0")
    ax_delta.set_title("Change in similarity (target in teal)", loc="left", fontsize=11)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(labels, rotation=20, ha="right")
    for i, delta in enumerate(deltas):
        if pd.notna(delta):
            ax_delta.text(
                i,
                delta,
                f"{delta:+.2f}",
                ha="center",
                va="bottom" if delta >= 0 else "top",
                fontsize=8,
                fontweight="bold" if PARTY_ORDER[i] == steered else "normal",
            )

    fig.tight_layout()
    dest = _model_out(model) / "by_alpha" / _alpha_tag(alpha)
    dest.mkdir(parents=True, exist_ok=True)
    _save(fig, dest / f"steer_{steered}.png")


def plot_score_profile(model: str, table: pd.DataFrame, steered: str) -> None:
    sub = table[(table["model"] == model) & (table["steered_party_key"] == steered)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for color, pk in zip(PARTY_COLORS, PARTY_ORDER):
        party_rows = sub[sub["compared_party_key"] == pk].set_index("alpha").reindex(LINE_ALPHAS)
        ax.plot(
            [f"{a:g}" for a in LINE_ALPHAS],
            party_rows["mean_score"].to_numpy(dtype=float),
            marker="o",
            markersize=4,
            label=PARTIES[pk],
            color=color,
            linewidth=2.5 if pk == steered else 1.2,
            linestyle="-" if pk == steered else "--",
        )
    ax.set_ylim(0, 10)
    ax.set_title(
        f"{model_label(model)} — steering as {PARTIES[steered]}\n"
        "Cross-party similarity vs α (same α at each x)",
        fontweight="bold",
    )
    ax.set_xlabel("Steering α")
    ax.set_ylabel("Mean 1–10 similarity")
    ax.legend(title="Judged vs party", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, _model_out(model) / f"score_profile_{steered}.png")


def plot_model_alpha_bars(model: str, frame: pd.DataFrame, alpha: float, ylim: Tuple[float, float]) -> None:
    """Five steered-party panels, five judged-party bars — one α."""
    fig, axes = _grid(len(PARTY_ORDER), cell=(4.4, 3.8), max_cols=5)
    fig.suptitle(
        f"{model_label(model)} — {_alpha_title(alpha)} — cross-party Δ when steering as each party\n"
        "Same α in every panel; filled bar = steered (target) party",
        fontsize=13,
        fontweight="bold",
    )
    x = np.arange(len(PARTY_ORDER))
    for ax, steered in zip(axes, PARTY_ORDER):
        deltas = _judged_vector(frame, steered, "mean_delta").to_numpy(dtype=float)
        colors = [PARTY_COLORS[i] for i in range(len(PARTY_ORDER))]
        bars = ax.bar(x, deltas, color=colors, edgecolor="white")
        bars[PARTY_ORDER.index(steered)].set_linewidth(1.8)
        bars[PARTY_ORDER.index(steered)].set_edgecolor("black")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels([PARTIES[k] for k in PARTY_ORDER], rotation=30, ha="right")
        layer_rows = frame[frame["steered_party_key"] == steered]
        layer = int(layer_rows["layer"].iloc[0]) if not layer_rows.empty else None
        title = f"Steer {PARTIES[steered]}"
        if layer is not None:
            title += f"  (L{layer})"
        ax.set_title(title, fontweight="bold")
        if steered == PARTY_ORDER[0]:
            ax.set_ylabel("Mean Δ vs α=0")
        for i, delta in enumerate(deltas):
            if pd.notna(delta):
                ax.text(
                    i,
                    delta,
                    f"{delta:.2f}",
                    ha="center",
                    va="bottom" if delta >= 0 else "top",
                    fontsize=7,
                    fontweight="bold" if PARTY_ORDER[i] == steered else "normal",
                )
    _hide_extra(axes, len(PARTY_ORDER))
    fig.tight_layout()
    dest = _model_out(model) / "by_alpha" / _alpha_tag(alpha)
    dest.mkdir(parents=True, exist_ok=True)
    _save(fig, dest / "cross_party_bars.png")


def plot_model_alpha_gain_matrix(model: str, frame: pd.DataFrame, alpha: float, vmax: float) -> None:
    mat = pd.DataFrame(index=PARTY_ORDER, columns=PARTY_ORDER, dtype=float)
    for steered in PARTY_ORDER:
        mat.loc[steered] = _judged_vector(frame, steered, "mean_delta").to_numpy(dtype=float)
    mat.index = [PARTIES[k] for k in PARTY_ORDER]
    mat.columns = [PARTIES[k] for k in PARTY_ORDER]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sns.heatmap(
        mat.astype(float),
        ax=ax,
        cmap="RdYlGn",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Mean Δ vs reused α=0"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_title(
        f"{model_label(model)} — {_alpha_title(alpha)}\n"
        "Rows = steered as · Cols = judged vs",
        fontweight="bold",
    )
    ax.set_xlabel("Judged vs party")
    ax.set_ylabel("Steered as party")
    fig.tight_layout()
    dest = _model_out(model) / "by_alpha" / _alpha_tag(alpha)
    dest.mkdir(parents=True, exist_ok=True)
    _save(fig, dest / "gain_matrix.png")


def plot_alpha_gain_overview(table: pd.DataFrame, models: Sequence[str], alpha: float, vmax: float) -> None:
    """All models' steered×judged matrices at one α."""
    fig, axes = _grid(len(models), cell=(5.0, 4.2), max_cols=4)
    fig.suptitle(
        f"{_alpha_title(alpha)} — cross-party gain (steered × judged)\n"
        "Same α in every panel; diagonal = target-party Δ",
        fontsize=13,
        fontweight="bold",
    )
    frame = _slice_alpha(table, alpha)
    for ax, model in zip(axes, models):
        sub = frame[frame["model"] == model]
        mat = pd.DataFrame(index=PARTY_ORDER, columns=PARTY_ORDER, dtype=float)
        for steered in PARTY_ORDER:
            mat.loc[steered] = _judged_vector(sub, steered, "mean_delta").to_numpy(dtype=float)
        mat.index = [PARTIES[k] for k in PARTY_ORDER]
        mat.columns = [PARTIES[k] for k in PARTY_ORDER]
        sns.heatmap(
            mat.astype(float),
            ax=ax,
            cmap="RdYlGn",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            cbar=ax is axes[len(models) - 1],
            cbar_kws={"label": "Mean Δ"} if ax is axes[len(models) - 1] else None,
        )
        ax.set_title(model_label(model), fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Judged vs")
        ax.set_ylabel("Steered as")
    _hide_extra(axes, len(models))
    fig.tight_layout()
    dest = _out() / "by_alpha" / _alpha_tag(alpha)
    dest.mkdir(parents=True, exist_ok=True)
    _save(fig, dest / "gain_matrices_all_models.png")


def plot_alpha_steered_panels(table: pd.DataFrame, models: Sequence[str], alpha: float, steered: str, ylim: Tuple[float, float]) -> None:
    """One steered party, every model, judged-party bars — fixed α."""
    fig, axes = _grid(len(models), cell=(4.2, 3.5), max_cols=4)
    fig.suptitle(
        f"{_alpha_title(alpha)} — steering as {PARTIES[steered]}: effect on all judged parties\n"
        "Same α in every panel; filled outline = target party",
        fontsize=13,
        fontweight="bold",
    )
    frame = _slice_alpha(table, alpha)
    x = np.arange(len(PARTY_ORDER))
    for ax, model in zip(axes, models):
        sub = frame[frame["model"] == model]
        deltas = _judged_vector(sub, steered, "mean_delta").to_numpy(dtype=float)
        bars = ax.bar(x, deltas, color=PARTY_COLORS, edgecolor="white")
        bars[PARTY_ORDER.index(steered)].set_edgecolor("black")
        bars[PARTY_ORDER.index(steered)].set_linewidth(1.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels([PARTIES[k] for k in PARTY_ORDER], rotation=30, ha="right")
        ax.set_title(model_label(model), fontweight="bold")
        if model == models[0] or list(axes).index(ax) % 4 == 0:
            ax.set_ylabel("Mean Δ vs α=0")
    _hide_extra(axes, len(models))
    fig.tight_layout()
    dest = _out() / "by_alpha" / _alpha_tag(alpha)
    dest.mkdir(parents=True, exist_ok=True)
    _save(fig, dest / f"steer_{steered}_all_models.png")


def main() -> None:
    models = available_models(require_steering=True)
    if not models:
        print("Skipping cross-party profiles: no sweep summaries found.")
        return
    table = load_cross_party_fixed_alpha_table()
    csv_path = _out() / "cross_party_by_alpha.csv"
    table.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    for model in models:
        model_rows = table[(table["model"] == model) & (table["alpha"] > 0)]
        delta_ylim = _shared_ylim(model_rows["mean_delta"].to_numpy(dtype=float))
        print(f"Separate cross-party plots: {model_label(model)}")
        for alpha in SNAPSHOT_ALPHAS:
            for steered in PARTY_ORDER:
                plot_one_steered_party(model, steered, alpha, table, delta_ylim)


if __name__ == "__main__":
    main()
