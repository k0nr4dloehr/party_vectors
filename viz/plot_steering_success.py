"""Phase 3 judge-based steering success at a fixed injection strength.

Every comparison plot holds α constant so parties (and models) are comparable.
At a given α the layer with the largest target-party mean Δ among the three
probe-selected layers is used. Mixing selected configs across different α
values is intentionally avoided.

Metric: paired mean Δ similarity to the steered party vs the reused α=0
response on the selection set. Llama 3 70B often has n_pairs << 57.
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
    available_steering_parties,
    load_all_summaries,
    load_fixed_alpha_success_table,
    model_label,
    output_dir,
)

sns.set_theme(style="whitegrid", context="paper")

MODEL_PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
]

# α=0 is the baseline (Δ=0 by construction) and is omitted from comparisons.
COMPARISON_ALPHAS = tuple(a for a in ALPHA_GRID if a > 0)
PARTY_COLORS = sns.color_palette("Set2", n_colors=len(PARTY_ORDER))


def _out() -> Path:
    path = output_dir() / "steering_success"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _alpha_dir(alpha: float) -> Path:
    path = _out() / "by_alpha" / _alpha_tag(alpha)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_out(model: str) -> Path:
    path = _out() / model
    path.mkdir(parents=True, exist_ok=True)
    return path


def _alpha_tag(alpha: float) -> str:
    text = f"{alpha:g}"
    return "alpha_" + text.replace(".", "p")


def _alpha_title(alpha: float) -> str:
    return f"α = {alpha:g}"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _model_colors(models: Sequence[str]) -> dict:
    return {model: MODEL_PALETTE[i % len(MODEL_PALETTE)] for i, model in enumerate(models)}


def _grid(n: int, cell: Tuple[float, float] = (5.6, 4.6), max_cols: int = 4):
    ncols = min(max_cols, max(n, 1))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows), squeeze=False)
    return fig, axes.flatten(), nrows, ncols


def _hide_extra(axes: Iterable[plt.Axes], used: int) -> None:
    for ax in list(axes)[used:]:
        ax.set_visible(False)


def _slice_alpha(table: pd.DataFrame, alpha: float) -> pd.DataFrame:
    return table[np.isclose(table["alpha"], alpha)].copy()


def _pivot(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    mat = frame.pivot(index="model_label", columns="steered_party", values=value)
    model_labels = list(dict.fromkeys(frame["model_label"]))
    party_labels = [PARTIES[k] for k in PARTY_ORDER]
    return mat.reindex(index=model_labels, columns=party_labels)


def _shared_ylim(values: np.ndarray, pad: float = 0.12) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-0.5, 1.0)
    lo = min(0.0, float(np.min(finite)))
    hi = max(0.0, float(np.max(finite)))
    span = max(hi - lo, 0.4)
    return lo - pad * span, hi + pad * span


def _caption(ax: plt.Axes, alpha: float) -> None:
    ax.set_title(
        f"{_alpha_title(alpha)} — target-party mean Δ vs reused α=0\n"
        "Best of the 3 probe-selected layers; α held fixed so parties are comparable",
        fontweight="bold",
    )


def plot_alpha_heatmap(frame: pd.DataFrame, alpha: float, vmin: float, vmax: float) -> None:
    values = _pivot(frame, "mean_delta")
    stars = _pivot(frame, "ci_excludes_zero")
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    sns.heatmap(
        values.astype(float),
        ax=ax,
        cmap="RdYlGn",
        center=0,
        vmin=vmin,
        vmax=vmax,
        annot=False,
        cbar_kws={"label": "Mean Δ vs reused α=0 (1–10 similarity)"},
        linewidths=0.4,
        linecolor="white",
    )
    for i, idx in enumerate(values.index):
        for j, col in enumerate(values.columns):
            val = values.loc[idx, col]
            if pd.isna(val):
                continue
            mark = "*" if bool(stars.loc[idx, col]) else ""
            ax.text(j + 0.5, i + 0.5, f"{float(val):.2f}{mark}", ha="center", va="center", fontsize=8)
    ax.set_title(
        f"Steering gain at {_alpha_title(alpha)} (model × party)\n"
        "* = 95% CI excludes 0 · layer = best of 3 probe-selected layers",
        fontweight="bold",
    )
    ax.set_xlabel("Steered party")
    ax.set_ylabel("Model")
    fig.tight_layout()
    _save(fig, _alpha_dir(alpha) / "model_party_heatmap.png")


def plot_alpha_party_bars(frame: pd.DataFrame, alpha: float, models: Sequence[str], ylim: Tuple[float, float]) -> None:
    """Parties on x, one bar per model — compare parties at this α."""
    colors = _model_colors(models)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(PARTY_ORDER))
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        sub = frame[frame["model"] == model].set_index("steered_party_key").reindex(PARTY_ORDER)
        offsets = x + (i - (len(models) - 1) / 2) * width
        ax.bar(
            offsets,
            sub["mean_delta"].to_numpy(dtype=float),
            width=width,
            label=model_label(model),
            color=colors[model],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([PARTIES[k] for k in PARTY_ORDER])
    ax.set_ylim(*ylim)
    ax.set_ylabel("Mean Δ similarity vs reused α=0")
    ax.set_xlabel("Steered party")
    _caption(ax, alpha)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, _alpha_dir(alpha) / "party_comparison_grouped.png")


def plot_alpha_model_panels(frame: pd.DataFrame, alpha: float, models: Sequence[str], ylim: Tuple[float, float]) -> None:
    """One panel per model, five party bars — party comparison inside each model."""
    fig, axes, _, _ = _grid(len(models), cell=(4.4, 3.7), max_cols=4)
    fig.suptitle(
        f"{_alpha_title(alpha)} — how well did each party steer, within a model\n"
        "Same α in every panel; bar = target-party Δ at the best of 3 probe-selected layers",
        fontsize=13,
        fontweight="bold",
    )
    for ax, model in zip(axes, models):
        sub = frame[frame["model"] == model].set_index("steered_party_key").reindex(PARTY_ORDER)
        deltas = sub["mean_delta"].to_numpy(dtype=float)
        ax.bar([PARTIES[k] for k in PARTY_ORDER], deltas, color=PARTY_COLORS, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(model_label(model), fontweight="bold")
        if model == models[0] or list(axes).index(ax) % 4 == 0:
            ax.set_ylabel("Mean Δ vs α=0")
        for x_i, (party, delta, n_pairs, incomplete) in enumerate(
            zip(PARTY_ORDER, deltas, sub["n_pairs"], sub["incomplete_pairs"])
        ):
            if pd.isna(delta):
                continue
            note = f"{delta:.2f}"
            if bool(incomplete):
                note += f"\nn={int(n_pairs)}"
            ax.text(x_i, delta + (0.03 if delta >= 0 else -0.03), note, ha="center", va="bottom" if delta >= 0 else "top", fontsize=7)
    _hide_extra(axes, len(models))
    fig.tight_layout()
    _save(fig, _alpha_dir(alpha) / "per_model_party_bars.png")


def plot_alpha_specificity(frame: pd.DataFrame, alpha: float, vmin: float, vmax: float) -> None:
    values = _pivot(frame, "specificity")
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    sns.heatmap(
        values.astype(float),
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Target Δ − mean Δ of the other four parties"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_title(
        f"Specificity at {_alpha_title(alpha)}\n"
        "Positive: the steered party gained more than the other judged parties",
        fontweight="bold",
    )
    ax.set_xlabel("Steered party")
    ax.set_ylabel("Model")
    fig.tight_layout()
    _save(fig, _alpha_dir(alpha) / "specificity_heatmap.png")


def plot_alpha_overview(table: pd.DataFrame) -> None:
    """All α heatmaps in one figure, shared color scale, for scanning the dose."""
    fig, axes, _, _ = _grid(len(COMPARISON_ALPHAS), cell=(5.2, 4.0), max_cols=3)
    vmax = max(1.0, float(np.nanmax(np.abs(table.loc[table["alpha"] > 0, "mean_delta"].to_numpy(dtype=float)))))
    fig.suptitle(
        "Target-party steering gain at each α (shared color scale)\n"
        "Each panel holds α fixed; cell = best of 3 probe-selected layers",
        fontsize=13,
        fontweight="bold",
    )
    for ax, alpha in zip(axes, COMPARISON_ALPHAS):
        frame = _slice_alpha(table, alpha)
        values = _pivot(frame, "mean_delta")
        sns.heatmap(
            values.astype(float),
            ax=ax,
            cmap="RdYlGn",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
            cbar=ax is axes[len(COMPARISON_ALPHAS) - 1],
            cbar_kws={"label": "Mean Δ"} if ax is axes[len(COMPARISON_ALPHAS) - 1] else None,
        )
        ax.set_title(_alpha_title(alpha), fontweight="bold")
        ax.set_xlabel("Party")
        ax.set_ylabel("Model")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    _hide_extra(axes, len(COMPARISON_ALPHAS))
    fig.tight_layout()
    _save(fig, _out() / "by_alpha" / "overview_heatmaps_all_alphas.png")


def plot_dose_response_fixed_alpha(table: pd.DataFrame, models: Sequence[str]) -> None:
    """Party lines vs α, using the best layer at each α — comparable at every x."""
    colors = _model_colors(models)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for model in models:
        sub = table[(table["model"] == model) & (table["alpha"] > 0)]
        curve = sub.groupby("alpha")["mean_delta"].mean().reindex(COMPARISON_ALPHAS)
        ax.plot(
            [f"{a:g}" for a in COMPARISON_ALPHAS],
            curve.to_numpy(dtype=float),
            marker="o",
            lw=1.8,
            color=colors[model],
            label=model_label(model),
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Steering α (held fixed; layer = best of 3 at that α)")
    ax.set_ylabel("Mean target-party Δ vs α=0 (average over parties)")
    ax.set_title("Dose–response with α aligned across parties and models", fontweight="bold")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, _out() / "comparison_dose_response_fixed_alpha.png")

    fig, axes, _, _ = _grid(len(PARTY_ORDER), cell=(5.0, 3.8), max_cols=5)
    fig.suptitle(
        "Dose–response per party, α aligned across models\n"
        "Each x-position is the same α; no mixing of selected configs",
        fontsize=13,
        fontweight="bold",
    )
    for ax, party in zip(axes, PARTY_ORDER):
        for model in models:
            sub = table[(table["model"] == model) & (table["steered_party_key"] == party) & (table["alpha"] > 0)]
            sub = sub.set_index("alpha").reindex(COMPARISON_ALPHAS)
            ax.plot(
                [f"{a:g}" for a in COMPARISON_ALPHAS],
                sub["mean_delta"].to_numpy(dtype=float),
                marker="o",
                lw=1.4,
                color=colors[model],
                label=model_label(model),
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(PARTIES[party], fontweight="bold")
        ax.set_xlabel("α")
        if party == PARTY_ORDER[0]:
            ax.set_ylabel("Mean Δ vs α=0")
    axes[0].legend(fontsize=6, loc="upper left")
    _hide_extra(axes, len(PARTY_ORDER))
    fig.tight_layout()
    _save(fig, _out() / "comparison_dose_response_by_party.png")


def plot_per_model_alpha_party_grid(model: str, table: pd.DataFrame, ylim: Tuple[float, float]) -> None:
    """One subplot per α: five party bars. Parties are comparable inside each panel."""
    fig, axes, _, _ = _grid(len(COMPARISON_ALPHAS), cell=(4.6, 3.6), max_cols=3)
    fig.suptitle(
        f"{model_label(model)} — party steering success at each α\n"
        "Each panel is one α; compare the five parties inside a panel, not across panels with different α",
        fontsize=13,
        fontweight="bold",
    )
    model_rows = table[table["model"] == model]
    for ax, alpha in zip(axes, COMPARISON_ALPHAS):
        sub = _slice_alpha(model_rows, alpha).set_index("steered_party_key").reindex(PARTY_ORDER)
        deltas = sub["mean_delta"].to_numpy(dtype=float)
        ax.bar([PARTIES[k] for k in PARTY_ORDER], deltas, color=PARTY_COLORS, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(_alpha_title(alpha), fontweight="bold")
        if alpha in (COMPARISON_ALPHAS[0], COMPARISON_ALPHAS[3]):
            ax.set_ylabel("Mean Δ vs α=0")
        for x_i, delta in enumerate(deltas):
            if pd.notna(delta):
                ax.text(x_i, delta, f"{delta:.2f}", ha="center", va="bottom" if delta >= 0 else "top", fontsize=7)
    _hide_extra(axes, len(COMPARISON_ALPHAS))
    fig.tight_layout()
    _save(fig, _model_out(model) / "party_success_one_panel_per_alpha.png")


def plot_per_model_dose_response(model: str, table: pd.DataFrame) -> None:
    sub = table[(table["model"] == model) & (table["alpha"] > 0)]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for color, party in zip(PARTY_COLORS, PARTY_ORDER):
        party_rows = sub[sub["steered_party_key"] == party].set_index("alpha").reindex(COMPARISON_ALPHAS)
        ax.plot(
            [f"{a:g}" for a in COMPARISON_ALPHAS],
            party_rows["mean_delta"].to_numpy(dtype=float),
            marker="o",
            lw=2,
            color=color,
            label=PARTIES[party],
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Steering α (same α for every party at each x)")
    ax.set_ylabel("Mean target-party Δ vs reused α=0")
    ax.set_title(
        f"{model_label(model)} — party dose–response at aligned α\n"
        "Layer = best of 3 probe-selected layers at that α",
        fontweight="bold",
    )
    ax.legend(title="Steered party", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, _model_out(model) / "dose_response_aligned_alpha.png")


def plot_per_model_target_heatmaps(model: str) -> None:
    summaries = load_all_summaries([model])
    parties = available_steering_parties(model)
    fig, axes, _, _ = _grid(len(parties), cell=(4.4, 3.6), max_cols=5)
    fig.suptitle(
        f"{model_label(model)} — target-party Δ vs reused α=0 (layer × α)",
        fontsize=13,
        fontweight="bold",
    )
    vmax = 1.5
    for party in parties:
        sub = summaries[(summaries["steered_party_key"] == party) & (summaries["compared_party_key"] == party)]
        if sub.empty:
            continue
        vmax = max(vmax, float(np.nanmax(np.abs(sub["mean_delta"].to_numpy(dtype=float)))))
    for ax, party in zip(axes, parties):
        sub = summaries[(summaries["steered_party_key"] == party) & (summaries["compared_party_key"] == party)]
        pivot = sub.pivot(index="layer", columns="alpha", values="mean_delta")
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdYlGn",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            cbar=ax is axes[len(parties) - 1],
            cbar_kws={"label": "Mean Δ"} if ax is axes[len(parties) - 1] else None,
        )
        ax.set_title(PARTIES[party], fontweight="bold")
        ax.set_xlabel("α")
        ax.set_ylabel("Layer")
    _hide_extra(axes, len(parties))
    fig.tight_layout()
    _save(fig, _model_out(model) / "target_delta_heatmaps.png")


def plot_fixed_alpha_comparisons(table: pd.DataFrame, models: Sequence[str]) -> None:
    nonzero = table[table["alpha"] > 0]
    heat_max = max(1.0, float(np.nanpercentile(np.abs(nonzero["mean_delta"].to_numpy(dtype=float)), 98)))
    spec_max = max(1.0, float(np.nanpercentile(np.abs(nonzero["specificity"].to_numpy(dtype=float)), 98)))
    for alpha in COMPARISON_ALPHAS:
        frame = _slice_alpha(table, alpha)
        if frame.empty:
            continue
        ylim = _shared_ylim(frame["mean_delta"].to_numpy(dtype=float))
        print(f"Fixed-alpha figures: alpha={alpha:g}")
        plot_alpha_heatmap(frame, alpha, vmin=-heat_max, vmax=heat_max)
        plot_alpha_party_bars(frame, alpha, models, ylim)
        plot_alpha_model_panels(frame, alpha, models, ylim)
        plot_alpha_specificity(frame, alpha, vmin=-spec_max, vmax=spec_max)
    plot_alpha_overview(table)
    plot_dose_response_fixed_alpha(table, models)


def plot_per_model(model: str, table: pd.DataFrame, ylim: Tuple[float, float]) -> None:
    plot_per_model_target_heatmaps(model)
    plot_per_model_dose_response(model, table)
    plot_per_model_alpha_party_grid(model, table, ylim)


def main() -> None:
    models = available_models(require_steering=True)
    if not models:
        print("Skipping steering-success plots: no sweep summaries found.")
        return
    table = load_fixed_alpha_success_table()
    csv_path = _out() / "steering_success_by_alpha.csv"
    table.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    plot_fixed_alpha_comparisons(table, models)
    for model in models:
        print(f"Per-model figures: {model_label(model)}")
        model_ylim = _shared_ylim(
            table.loc[(table["model"] == model) & (table["alpha"] > 0), "mean_delta"].to_numpy(dtype=float)
        )
        plot_per_model(model, table, model_ylim)


if __name__ == "__main__":
    main()
