"""Layer-trend view: target-party judge delta vs layer, per alpha.

For methodology design: shows whether any ideological signal is emerging with
depth in the all-positions/normalized-injection run, and how large alphas
affect raw judge scores (degradation check via mean raw sim at alpha=8).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARTIAL = ROOT / "data" / "results_new" / (sys.argv[1] if len(sys.argv) > 1 else "partial_allpos")
OUT = ROOT / "figures" / "partial_checkpoints" / PARTIAL.name
TAG = "meta-llama-3-8b-instruct"
PARTIES = ["CDU_CSU", "GRUENE", "SPD", "AfD", "DIE_LINKE"]
ALPHAS = [1.0, 2.0, 4.0, 8.0]


def load_party(party: str) -> pd.DataFrame:
    df = pd.read_csv(PARTIAL / f"sweep_ckpt_{TAG}_{party}.csv")
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sim_col = {p: f"sim_{p}" for p in PARTIES}

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5), sharey=True)
    for ax, party in zip(axes, PARTIES):
        df = load_party(party)
        base = df[df["alpha"] == 0.0].set_index(["layer", "thesis_key"])[sim_col[party]]
        df = df.copy()
        df["base_sim"] = df.set_index(["layer", "thesis_key"]).index.map(base)
        df["delta"] = df[sim_col[party]] - df["base_sim"]
        for alpha in ALPHAS:
            d = df[df["alpha"] == alpha].groupby("layer")["delta"]
            mean = d.mean()
            ci = 1.96 * d.std() / np.sqrt(d.count())
            ax.errorbar(mean.index, mean.values, yerr=ci.values, marker="o", ms=3,
                        lw=1.2, capsize=2, label=f"a={alpha:g}")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"steer {party}", fontweight="bold")
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("target-party judge delta")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("All-positions normalized injection — target delta vs layer (Llama-3-8B)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p1 = OUT / "layer_trend_deltas.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p1}")

    # Degradation check: raw mean target sim and response word count at high alpha
    rows = []
    for party in PARTIES:
        df = load_party(party)
        df["nwords"] = df["steered_response"].astype(str).str.split().str.len()
        for alpha in [0.0, 4.0, 8.0]:
            d = df[df["alpha"] == alpha]
            rows.append({
                "party": party, "alpha": alpha,
                "raw_target_sim": round(float(d[sim_col[party]].mean()), 3),
                "raw_max_sim_any_party": round(float(d[[sim_col[p] for p in PARTIES]].max(axis=1).mean()), 3),
                "mean_words": round(float(d["nwords"].mean()), 1),
            })
    tab = pd.DataFrame(rows).pivot(index="alpha", columns="party",
                                   values=["raw_target_sim", "raw_max_sim_any_party", "mean_words"])
    print("\n=== degradation check (raw means by alpha) ===")
    print(tab.to_string())


if __name__ == "__main__":
    main()
