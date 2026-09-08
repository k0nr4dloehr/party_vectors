"""Cosine similarity between party vectors + norms, per layer.

Diagnostics: if party vectors are highly collinear (cos ~ 1), they encode a
shared direction (e.g. partisan style) rather than party-specific ideology.
Also reports vector norms per layer (injection scale question).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

PARTIES = ["CDU_CSU", "GRUENE", "SPD", "AfD", "DIE_LINKE"]
TAG = "meta-llama-3-8b-instruct"

ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "data" / "vectors_new" / TAG
OUT = ROOT / "figures" / "vector_diagnostics"


def load_vectors(party: str, base: Path | None = None) -> dict[int, torch.Tensor]:
    base = base or VEC
    meta = json.loads((base / party / "metadata.json").read_text(encoding="utf-8"))
    n = meta["num_layers"]
    return {
        layer: torch.load(base / party / f"vector_layer_{layer:02d}.pt", weights_only=True)
        for layer in range(n)
    }


def main() -> None:
    import sys

    base = Path(sys.argv[1]) if len(sys.argv) > 1 else VEC
    out_dir = OUT / base.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    vecs = {p: load_vectors(p, base) for p in PARTIES}
    num_layers = len(vecs[PARTIES[0]])

    norms = pd.DataFrame({p: {l: float(v[l].norm()) for l in range(num_layers)} for p, v in vecs.items()})
    norms.index.name = "layer"

    cos_mean, cos_min, cos_max = [], [], []
    mats: dict[int, np.ndarray] = {}
    for layer in range(num_layers):
        V = torch.stack([vecs[p][layer] for p in PARTIES]).float()
        V = V / V.norm(dim=1, keepdim=True)
        C = (V @ V.T).numpy()
        mats[layer] = C
        off = C[~np.eye(len(PARTIES), dtype=bool)]
        cos_mean.append(off.mean())
        cos_min.append(off.min())
        cos_max.append(off.max())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    for p in PARTIES:
        ax.plot(norms.index, norms[p], label=p, marker="o", ms=3)
    ax.set_title("Vector L2 norms per layer", fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("||v||")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(range(num_layers), cos_mean, color="black", lw=2, label="mean pairwise cos")
    ax.fill_between(range(num_layers), cos_min, cos_max, alpha=0.3, label="min-max")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("Cross-party cosine similarity per layer", fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(v_i, v_j)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p1 = out_dir / "vector_norms_cosines.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p1}")

    sample_layers = [0, 6, 12, 18, 24, 31]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, layer in zip(axes.flatten(), sample_layers):
        sns.heatmap(
            mats[layer],
            ax=ax,
            xticklabels=PARTIES,
            yticklabels=PARTIES,
            vmin=-1,
            vmax=1,
            center=0,
            cmap="RdBu_r",
            annot=True,
            fmt=".2f",
        )
        ax.set_title(f"Layer {layer}", fontweight="bold")
    fig.suptitle("Party-vector cosine matrices (prompt-last-token vectors)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p2 = out_dir / "vector_cosine_matrices.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p2}")

    print("\n=== norms (selected layers) ===")
    print(norms.loc[sample_layers].round(2).to_string())
    print("\n=== mean pairwise cosine per layer ===")
    for layer in range(num_layers):
        print(f"  L{layer:02d}: mean={cos_mean[layer]:+.3f}  range=[{cos_min[layer]:+.3f}, {cos_max[layer]:+.3f}]  norm~{norms.loc[layer].mean():.2f}")


if __name__ == "__main__":
    main()
