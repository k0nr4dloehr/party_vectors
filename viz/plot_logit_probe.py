"""Visualize Phase 3b logit-probe results.

Two views per party:
1. Heatmap (layer x alpha) of the mean congruent-margin shift vs alpha=0 —
   where in depth and at what strength the vector moves the party-congruent
   stance token's probability.
2. Line plot of argmax3 flip rate (fraction of theses whose top candidate
   becomes the party-congruent stance) vs layer, per alpha.

Usage: python viz/plot_logit_probe.py [results_subdir]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results_new" / (sys.argv[1] if len(sys.argv) > 1 else "probe")
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "selection"
TAG = sys.argv[3] if len(sys.argv) > 3 else "meta-llama-3-8b-instruct"
OUT = ROOT / "figures" / "logit_probe" / TAG
PARTIES = ["CDU_CSU", "GRUENE", "SPD", "AfD", "DIE_LINKE"]


def load_party(party: str) -> pd.DataFrame:
    candidates = [
        RESULTS / f"probe_results_{TAG}_{party}_{SPLIT}.csv",
        RESULTS / f"probe_ckpt_{TAG}_{party}_{SPLIT}.csv",
        RESULTS / f"probe_results_{TAG}_{party}.csv",
        RESULTS / f"probe_ckpt_{TAG}_{party}.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(candidates[0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(26, 5), sharey=True)
    summaries = []
    for ax, party in zip(axes, PARTIES):
        df = load_party(party)
        piv = df.pivot_table(index=["layer", "thesis_key"], columns="alpha", values="margin_cong")
        alphas = [a for a in piv.columns if a != 0.0]
        mat = pd.DataFrame({
            a: (piv[a] - piv[0.0]).groupby("layer").mean() for a in alphas
        })
        sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0, annot=False,
                    cbar=ax is axes[-1], cbar_kws={"label": "mean margin shift"})
        ax.set_title(f"steer {party}", fontweight="bold")
        ax.set_xlabel("alpha")
        if ax is axes[0]:
            ax.set_ylabel("Layer")
        best = mat.stack().idxmax(), float(mat.stack().max())
        worst = mat.stack().idxmin(), float(mat.stack().min())
        summaries.append((party, best, worst))
    fig.suptitle("Logit probe: congruent-stance margin shift (all-positions, unit-norm injection)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p1 = OUT / f"probe_margin_heatmaps_{SPLIT}.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p1}")

    fig, axes = plt.subplots(1, 5, figsize=(26, 4.5), sharey=True)
    for ax, party in zip(axes, PARTIES):
        df = load_party(party)
        base = df[df["alpha"] == 0.0].groupby("layer")["argmax3_is_cong"].mean()
        for alpha, g in df[df["alpha"] != 0.0].groupby("alpha"):
            rate = g.groupby("layer")["argmax3_is_cong"].mean()
            ax.plot(rate.index, rate.values - base.reindex(rate.index).values,
                    marker="o", ms=2.5, lw=1.1, label=f"a={alpha:g}")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"steer {party}", fontweight="bold")
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("delta flip rate (congruent argmax)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Logit probe: change in fraction of theses whose top-1 stance is party-congruent",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p2 = OUT / f"probe_fliprate_curves_{SPLIT}.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p2}")

    print("\n=== best / worst (layer, alpha) by mean margin shift ===")
    for party, best, worst in summaries:
        print(f"  {party}: best {best[0]} -> {best[1]:+.3f} | worst {worst[0]} -> {worst[1]:+.3f}")

    print("\n=== baseline (alpha=0) argmax-congruent rate per party ===")
    for party in PARTIES:
        df = load_party(party)
        rate = df[df["alpha"] == 0.0]["argmax3_is_cong"].mean()
        p3 = df[df["alpha"] == 0.0]["p3_cong"].mean()
        print(f"  {party}: argmax-congruent {rate:.2f}, mean p3_cong {p3:.2f}")


def success_rate_views() -> None:
    """Raw per-layer success rate: fraction of theses whose top answer among the
    three candidate stance tokens (argmax3) is the party-congruent stance."""
    fig, axes = plt.subplots(1, 5, figsize=(26, 4.5), sharey=True)
    for ax, party in zip(axes, PARTIES):
        df = load_party(party).copy()
        for alpha, g in df.groupby("alpha"):
            rate = g.groupby("layer")["argmax3_is_cong"].mean()
            is_base = alpha == 0.0
            ax.plot(rate.index, rate.values, marker="o", ms=2.5,
                    lw=2.0 if is_base else 1.1,
                    color="black" if is_base else None,
                    ls="--" if is_base else "-",
                    label=f"a={alpha:g}")
        ax.set_title(f"steer {party}", fontweight="bold")
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("success rate (argmax over 3 stance tokens)")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"Logit probe success rate per layer ({SPLIT} split, {TAG}): fraction of "
                 f"theses whose most likely stance token aligns with the steered party",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p3 = OUT / f"probe_successrate_{SPLIT}.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p3}")


if __name__ == "__main__":
    main()
    success_rate_views()
