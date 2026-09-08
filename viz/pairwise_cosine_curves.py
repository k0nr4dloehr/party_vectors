"""Pairwise cosine similarity between party vectors, per layer.

Two views, original and cross-party-centered vectors:
  1. pair x layer heatmaps
  2. per-pair cosine curves across layers
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vector_cosine_analysis import PARTIES, load_vectors  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "vector_diagnostics"

PAIRS = list(itertools.combinations(range(len(PARTIES)), 2))
PAIR_LABELS = [f"{PARTIES[i]}–{PARTIES[j]}" for i, j in PAIRS]


def pairwise_cosines(V: torch.Tensor) -> list[float]:
    Vn = V / V.norm(dim=1, keepdim=True).clamp(min=1e-12)
    C = Vn @ Vn.T
    return [float(C[i, j]) for i, j in PAIRS]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vecs = {p: load_vectors(p) for p in PARTIES}
    num_layers = len(vecs[PARTIES[0]])

    orig = np.zeros((len(PAIRS), num_layers))
    cent = np.zeros((len(PAIRS), num_layers))
    for layer in range(num_layers):
        V = torch.stack([vecs[p][layer] for p in PARTIES]).float()
        orig[:, layer] = pairwise_cosines(V)
        cent[:, layer] = pairwise_cosines(V - V.mean(dim=0, keepdim=True))

    # --- figure 1: pair x layer heatmaps ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for ax, mat, title in zip(
        axes,
        [orig, cent],
        ["Original vectors", "Centered vectors (cross-party mean removed)"],
    ):
        sns.heatmap(
            mat,
            ax=ax,
            yticklabels=PAIR_LABELS,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "cosine"},
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Layer")
        xticks = np.arange(0, num_layers, 2) + 0.5
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(i) for i in range(0, num_layers, 2)])
    fig.suptitle("Pairwise party-vector cosine similarity per layer", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p1 = OUT / "pairwise_cosine_heatmaps.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p1}")

    # --- figure 2: curves ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, mat, title in zip(
        axes,
        [orig, cent],
        ["Original vectors", "Centered vectors"],
    ):
        for k, label in enumerate(PAIR_LABELS):
            ax.plot(range(num_layers), mat[k], label=label, color=cmap(k), lw=1.6)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cosine similarity")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Pairwise cosine curves (10 party pairs)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p2 = OUT / "pairwise_cosine_curves.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p2}")

    # --- console summary: closest/most distant pairs at key layers ---
    for layer in [8, 12, 15, 20, 26, 31]:
        order = np.argsort(orig[:, layer])
        closest = PAIR_LABELS[order[-1]]
        distant = PAIR_LABELS[order[0]]
        print(
            f"L{layer:02d}  original: closest={closest} ({orig[order[-1], layer]:+.2f})  "
            f"most distant={distant} ({orig[order[0], layer]:+.2f})   |   "
            f"centered range [{cent[:, layer].min():+.2f}, {cent[:, layer].max():+.2f}]"
        )

    print("\ncentered cosines at L15 (peak distinctive layer):")
    order = np.argsort(cent[:, 15])
    for k in order:
        print(f"  {PAIR_LABELS[k]:22s} {cent[k, 15]:+.3f}")


if __name__ == "__main__":
    main()
