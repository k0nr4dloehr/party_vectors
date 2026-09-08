"""Partial-results visualization from Phase 3 sweep checkpoints.

Loads sweep_ckpt_*.csv + alpha0_ckpt_*.csv from data/results_new/partial/
(interrupted run: complete layers only) and plots paired similarity deltas
vs the reused alpha=0 baseline. Answers: did the new methodology steer?
"""

from __future__ import annotations

from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PARTIES = {
    "CDU_CSU": "CDU / CSU",
    "GRUENE": "GRÜNE",
    "SPD": "SPD",
    "AfD": "AfD",
    "DIE_LINKE": "DIE LINKE",
}
PARTY_ORDER = list(PARTIES.keys())
TAG = "meta-llama-3-8b-instruct"

ROOT = Path(__file__).resolve().parent.parent
PARTIAL = ROOT / "data" / "results_new" / (sys.argv[1] if len(sys.argv) > 1 else "partial")
OUT = ROOT / "figures" / "partial_checkpoints" / PARTIAL.name


def load_party_frames(party: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sweep = pd.read_csv(PARTIAL / f"sweep_ckpt_{TAG}_{party}.csv")
    alpha0 = pd.read_csv(PARTIAL / f"alpha0_ckpt_{TAG}_{party}.csv")
    return sweep, alpha0


def paired_deltas(sweep: pd.DataFrame, alpha0: pd.DataFrame, compared: str) -> pd.DataFrame:
    """Mean paired delta vs alpha=0 per (layer, alpha), plus bootstrap CI."""
    col = f"sim_{compared}"
    base = alpha0.set_index("thesis_key")[col].to_dict()
    rows = []
    for (layer, alpha), grp in sweep.groupby(["layer", "alpha"]):
        deltas = [
            float(row[col]) - float(base[row["thesis_key"]])
            for _, row in grp.iterrows()
            if row["thesis_key"] in base
            and not pd.isna(row[col])
            and not pd.isna(base[row["thesis_key"]])
        ]
        if not deltas:
            continue
        arr = np.asarray(deltas)
        if alpha == 0.0 or len(arr) < 2:
            ci_low = ci_high = 0.0 if alpha == 0.0 else (float(arr.mean()),) * 2
        else:
            rng = np.random.default_rng(0)
            boots = rng.choice(arr, size=(1000, len(arr)), replace=True).mean(axis=1)
            ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        rows.append(
            {
                "layer": int(layer),
                "alpha": float(alpha),
                "mean_delta": float(arr.mean()),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n": len(arr),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
    axes = axes.flatten()
    verdicts = []

    for i, steered in enumerate(PARTY_ORDER):
        sweep, alpha0 = load_party_frames(steered)
        layers_done = sorted(sweep["layer"].unique())
        deltas = paired_deltas(sweep, alpha0, steered)
        nz = deltas[deltas["alpha"] != 0.0]

        if nz.empty:
            print(f"{steered}: layers {layers_done} — no nonzero-alpha data yet")
            continue

        best = nz.loc[nz["mean_delta"].idxmax()]
        frac_pos = float((nz["mean_delta"] > 0).mean())
        sig = nz[nz["ci_low"] > 0]
        verdicts.append(
            {
                "steered": steered,
                "layers_done": f"{len(layers_done)} ({layers_done[0]}-{layers_done[-1]})",
                "best_layer": int(best["layer"]),
                "best_alpha": best["alpha"],
                "best_delta": round(best["mean_delta"], 3),
                "ci": f"[{best['ci_low']:.2f}, {best['ci_high']:.2f}]",
                "frac_pos": f"{frac_pos:.0%}",
                "sig_pos_configs": len(sig),
            }
        )

        ax = axes[i]
        pivot = deltas.pivot(index="layer", columns="alpha", values="mean_delta")
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdYlGn",
            center=0,
            vmin=-1.5,
            vmax=1.5,
            cbar_kws={"label": "Δ similarity vs α=0"},
        )
        ax.set_title(f"Steer as {PARTIES[steered]} → sim to {PARTIES[steered]}", fontweight="bold")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Layer")

    for ax in axes[len(PARTY_ORDER) :]:
        ax.set_visible(False)

    fig.suptitle(
        "PARTIAL sweep (checkpoint, Llama 3 8B) — prompt-last-token vectors, "
        "site-matched injection\nTarget-party similarity delta, selection set",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT / "partial_target_heatmaps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    table = pd.DataFrame(verdicts)
    print("\n=== VERDICT TABLE (target party delta, partial layers) ===")
    print(table.to_string(index=False))
    table.to_csv(OUT / "partial_verdicts.csv", index=False)

    # specificity: at each party's best config, deltas across all compared parties
    spec_rows = []
    for v in verdicts:
        steered = v["steered"]
        sweep, alpha0 = load_party_frames(steered)
        sub = sweep[(sweep["layer"] == v["best_layer"]) & (sweep["alpha"] == v["best_alpha"])]
        for pk in PARTY_ORDER:
            col = f"sim_{pk}"
            base = alpha0.set_index("thesis_key")[col].to_dict()
            d = [
                float(row[col]) - float(base[row["thesis_key"]])
                for _, row in sub.iterrows()
                if row["thesis_key"] in base and not pd.isna(row[col]) and not pd.isna(base[row["thesis_key"]])
            ]
            spec_rows.append(
                {"steered": steered, "compared": pk, "delta": float(np.mean(d)) if d else np.nan}
            )
    spec = pd.DataFrame(spec_rows)
    mat = spec.pivot(index="steered", columns="compared", values="delta").loc[PARTY_ORDER, PARTY_ORDER]

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1.5,
        vmax=1.5,
        ax=ax2,
        cbar_kws={"label": "Δ similarity"},
    )
    ax2.set_title(
        "Specificity at best config per party (partial)\nrows = steered party, cols = judged party",
        fontweight="bold",
    )
    fig2.tight_layout()
    path2 = OUT / "partial_specificity_matrix.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {path2}")


if __name__ == "__main__":
    main()
