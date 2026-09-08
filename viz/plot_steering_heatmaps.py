"""Layer × alpha heatmaps of steering effect (delta vs reused α=0)."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data_utils import MODELS, PARTIES, PARTY_ORDER, available_models, load_all_summaries, output_dir


def plot_model_heatmaps(model: str) -> None:
    summaries = load_all_summaries([model])
    model_label = MODELS[model]
    out = output_dir() / "steering_heatmaps" / model
    out.mkdir(parents=True, exist_ok=True)

    for steered in PARTY_ORDER:
        sub = summaries[summaries["steered_party_key"] == steered]
        selection_sub = sub[sub["split"] == "selection"]
        if selection_sub.empty:
            selection_sub = sub
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        fig.suptitle(
            f"{model_label} — steering as {PARTIES[steered]} (selection set)\n"
            "Mean similarity delta vs reused α=0",
            fontsize=14,
            fontweight="bold",
        )

        for ax, pk in zip(axes, PARTY_ORDER):
            party_sub = selection_sub[selection_sub["compared_party_key"] == pk]
            delta_col = "mean_delta" if "mean_delta" in party_sub.columns else "delta_vs_alpha0"
            pivot = party_sub.pivot(index="layer", columns="alpha", values=delta_col)
            vals = np.abs(pivot.to_numpy(dtype=float))
            vmax = max(1.5, float(np.nanmax(vals))) if vals.size else 1.5
            sns.heatmap(
                pivot,
                ax=ax,
                cmap="RdYlGn",
                center=0,
                cbar_kws={"label": "Δ similarity"},
                vmin=-vmax,
                vmax=vmax,
            )
            title = PARTIES[pk]
            if pk == steered:
                title += " (target)"
            ax.set_title(title, fontweight="bold" if pk == steered else "normal")
            ax.set_xlabel("Alpha")
            ax.set_ylabel("Layer")

        for ax in axes[len(PARTY_ORDER) :]:
            ax.set_visible(False)

        fig.tight_layout()
        fig.savefig(out / f"heatmap_{steered}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out / f'heatmap_{steered}.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Single model short name")
    args = parser.parse_args()
    models = [args.model] if args.model else available_models(require_steering=True)
    for model in models:
        plot_model_heatmaps(model)


if __name__ == "__main__":
    main()
