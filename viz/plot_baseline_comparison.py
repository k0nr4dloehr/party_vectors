"""Compare party-prompted baseline, neutral baseline, unsteered, and best steered scores."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_utils import MODELS, PARTIES, PARTY_ORDER, available_models, build_comparison_table, output_dir


def _safe_float(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if pd.isna(value):
        return np.nan
    return float(value)


def plot_baseline_bars(model: str) -> None:
    df = build_comparison_table(model)
    model_label = MODELS[model]
    out = output_dir() / "baseline_comparison"
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(PARTY_ORDER), figsize=(20, 5), sharey=True)
    fig.suptitle(
        f"{model_label} — Target-party similarity by generation method",
        fontsize=14,
        fontweight="bold",
    )

    for ax, steered in zip(axes, PARTY_ORDER):
        sub = df[(df["steered_party_key"] == steered) & (df["compared_party_key"] == steered)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        method_values = [
            ("Party-prompted", _safe_float(row["party_prompted_baseline"]), "#4C72B0"),
            ("Neutral", _safe_float(row["neutral_baseline"]), "#55A868"),
            ("Unsteered α=0", _safe_float(row["steering_alpha0"]), "#C44E52"),
            ("Best steered (val)", _safe_float(row["best_steered_score"]), "#8172B2"),
        ]
        methods = [name for name, val, _ in method_values if not np.isnan(val)]
        values = [val for _, val, _ in method_values if not np.isnan(val)]
        colors = [color for _, val, color in method_values if not np.isnan(val)]
        bars = ax.bar(methods, values, color=colors, edgecolor="white")
        ax.set_title(PARTIES[steered], fontweight="bold")
        ax.set_ylim(0, 10)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15, f"{val:.1f}", ha="center", fontsize=9)
        if not np.isnan(_safe_float(row.get("best_steered_delta"))):
            ax.text(
                0.02,
                0.95,
                f"Δ={row['best_steered_delta']:+.2f}\nL{row['best_layer']} α{row['best_alpha']}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
        if steered == PARTY_ORDER[0]:
            ax.set_ylabel("Mean 1–10 similarity")
        ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    path = out / f"baseline_vs_steering_{model}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    fig, axes = plt.subplots(len(PARTY_ORDER), 1, figsize=(14, 3 * len(PARTY_ORDER)), sharex=True)
    if len(PARTY_ORDER) == 1:
        axes = [axes]
    for ax, steered in zip(axes, PARTY_ORDER):
        sub = df[df["steered_party_key"] == steered]
        plot_rows = []
        for _, row in sub.iterrows():
            plot_rows.append(
                {
                    "Judged vs": row["compared_party"],
                    "Score": row["steering_alpha0"],
                    "Reference": "Unsteered (α=0)",
                }
            )
            plot_rows.append(
                {
                    "Judged vs": row["compared_party"],
                    "Score": row["party_prompted_baseline"],
                    "Reference": "Party-prompted baseline",
                }
            )
        long_df = pd.DataFrame(plot_rows)
        sns.barplot(
            data=long_df,
            x="Judged vs",
            y="Score",
            hue="Reference",
            ax=ax,
            palette={"Unsteered (α=0)": "#C44E52", "Party-prompted baseline": "#4C72B0"},
        )
        ax.set_title(f"Steered as {PARTIES[steered]}", fontweight="bold")
        ax.set_ylim(0, 10)
    axes[-1].tick_params(axis="x", rotation=30)
    fig.suptitle(
        f"{model_label} — Unsteered steering vs party-prompted baseline by steered party",
        fontweight="bold",
    )
    fig.tight_layout()
    path2 = out / f"unsteered_vs_party_prompted_{model}.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path2}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    for model in ([args.model] if args.model else available_models(require_steering=True)):
        plot_baseline_bars(model)


if __name__ == "__main__":
    main()
