"""Summary matrices and bar charts of best steering gains per party."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_utils import (
    PARTIES,
    PARTY_ORDER,
    available_models,
    build_comparison_table,
    cross_party_gain_matrix,
    model_label,
    output_dir,
)


def plot_gain_matrices() -> None:
    out = output_dir() / "steering_summary"
    out.mkdir(parents=True, exist_ok=True)
    models = available_models(require_steering=True)
    if not models:
        print("Skipping gain matrices: no steering summaries found.")
        return

    n = len(models)
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.4 * nrows), squeeze=False)
    axes = axes.flatten()
    fig.suptitle(
        "Selection-set max steering gain (mean Δ vs reused α=0)\n"
        "Rows = steered party, Cols = judged party",
        fontsize=14,
        fontweight="bold",
    )

    for ax, model in zip(axes, models):
        mat = cross_party_gain_matrix(model, split="selection")
        sns.heatmap(
            mat.astype(float),
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            vmin=-1.5,
            vmax=2.5,
            cbar_kws={"label": "Max Δ"},
        )
        ax.set_title(model_label(model), fontweight="bold")
        ax.set_xlabel("Judged vs party")
        ax.set_ylabel("Steered as party")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    path = out / "cross_party_gain_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_target_party_gains() -> None:
    out = output_dir() / "steering_summary"
    models = available_models(require_steering=True)
    if not models:
        print("Skipping target-party gains: no steering summaries found.")
        return

    rows = []
    for model in models:
        comparison = build_comparison_table(model)
        for steered in PARTY_ORDER:
            sub = comparison[
                (comparison["steered_party_key"] == steered)
                & (comparison["compared_party_key"] == steered)
            ]
            if sub.empty:
                continue
            row = sub.iloc[0]
            rows.append(
                {
                    "model_label": model_label(model),
                    "steered_party": PARTIES[steered],
                    "party_prompted": row["party_prompted_baseline"],
                    "unsteered": row["steering_alpha0"],
                    "best_steered": row["best_steered_score"],
                    "best_delta": row["best_steered_delta"],
                    "best_ci_low": row.get("best_steered_ci_low"),
                    "best_ci_high": row.get("best_steered_ci_high"),
                    "best_layer": row["best_layer"],
                    "best_alpha": row["best_alpha"],
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        print("Skipping target-party gains: no comparison rows found.")
        return

    n = len(models)
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 5.2 * nrows), sharey=True, squeeze=False)
    axes = axes.flatten()
    fig.suptitle(
        "Target-party steering effectiveness vs baselines (selection-set config)",
        fontsize=14,
        fontweight="bold",
    )

    for ax, model in zip(axes, models):
        label = model_label(model)
        sub = df[df["model_label"] == label]
        x = np.arange(len(sub))
        width = 0.25
        ax.bar(x - width, sub["party_prompted"], width, label="Party-prompted", color="#4C72B0")
        ax.bar(x, sub["unsteered"], width, label="Reused α=0", color="#C44E52")
        ax.bar(x + width, sub["best_steered"], width, label="Best steered (selection)", color="#8172B2")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["steered_party"], rotation=30, ha="right")
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0, 10)
        for idx, row in sub.reset_index(drop=True).iterrows():
            if pd.isna(row["best_delta"]) or pd.isna(row["best_layer"]) or pd.isna(row["best_alpha"]):
                continue
            ax.annotate(
                f"Δ={row['best_delta']:+.2f}\nL{int(row['best_layer'])} α{row['best_alpha']:.1f}",
                xy=(idx + width, row["best_steered"] if pd.notna(row["best_steered"]) else 0),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        if model == models[0] or axes.tolist().index(ax) % ncols == 0:
            ax.set_ylabel("Mean similarity to own party reasoning")
        ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    path = out / "target_party_steering_gains.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    csv_path = out / "steering_effectiveness_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


def plot_model_comparison() -> None:
    out = output_dir() / "steering_summary"
    models = available_models(require_steering=True)
    if len(models) < 2:
        print("Skipping model comparison: need steering summaries for at least two models.")
        return

    rows = []
    for model in models:
        comparison = build_comparison_table(model)
        for steered in PARTY_ORDER:
            sub = comparison[
                (comparison["steered_party_key"] == steered)
                & (comparison["compared_party_key"] == steered)
            ]
            if sub.empty:
                continue
            row = sub.iloc[0]
            rows.append(
                {
                    "model_label": model_label(model),
                    "steered_party": PARTIES[steered],
                    "selection_delta": row["best_steered_delta"],
                }
            )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 6.2))
    sns.barplot(
        data=df,
        x="steered_party",
        y="selection_delta",
        hue="model_label",
        ax=ax,
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Selection-set steering gain by model", fontweight="bold")
    ax.set_xlabel("Steered as party")
    ax.set_ylabel("Mean Δ similarity vs reused α=0")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = out / "model_comparison_target_delta.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    plot_gain_matrices()
    plot_target_party_gains()
    plot_model_comparison()


if __name__ == "__main__":
    main()
