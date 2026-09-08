"""Centered-vector diagnostic: how much party-distinctive signal survives?

Per layer: subtract cross-party mean, report centered norms, distinctive
fraction ||v'||/||v||, and pairwise cosines after centering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viz"))
from vector_cosine_analysis import PARTIES, load_vectors  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures" / "vector_diagnostics"


def main() -> None:
    vecs = {p: load_vectors(p) for p in PARTIES}
    num_layers = len(vecs[PARTIES[0]])

    rows = []
    for layer in range(num_layers):
        V = torch.stack([vecs[p][layer] for p in PARTIES]).float()
        m = V.mean(dim=0, keepdim=True)
        Vc = V - m
        norms = V.norm(dim=1)
        cnorms = Vc.norm(dim=1)
        Cn = Vc / Vc.norm(dim=1, keepdim=True).clamp(min=1e-12)
        C = (Cn @ Cn.T).numpy()
        off = C[~np.eye(len(PARTIES), dtype=bool)]
        rows.append(
            {
                "layer": layer,
                "norm_mean": float(norms.mean()),
                "centered_norm_mean": float(cnorms.mean()),
                "distinctive_frac": float((cnorms / norms.clamp(min=1e-12)).mean()),
                "cos_mean_after": float(off.mean()),
                "cos_min_after": float(off.min()),
                "cos_max_after": float(off.max()),
                **{f"cnorm_{p}": float(cnorms[i]) for i, p in enumerate(PARTIES)},
            }
        )
    df = pd.DataFrame(rows).set_index("layer")
    print(df.round(3).to_string())
    df.to_csv(OUT / "centered_vector_profile.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    ax.plot(df.index, df["norm_mean"], label="original ||v||", color="gray")
    ax.plot(df.index, df["centered_norm_mean"], label="centered ||v'||", color="red", lw=2)
    for p in PARTIES:
        ax.plot(df.index, df[f"cnorm_{p}"], alpha=0.4, ls="--", label=f"v' {p}")
    ax.set_title("Vector norm before/after cross-party centering", fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 norm")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df.index, df["distinctive_frac"], color="darkgreen", lw=2)
    ax.set_title("Distinctive fraction ||v'||/||v|| per layer", fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("fraction")
    ax.grid(alpha=0.3)
    ax.axhline(0.2, color="red", ls=":", label="20% line")
    ax.legend()

    fig.tight_layout()
    out = OUT / "centered_norm_profile.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
