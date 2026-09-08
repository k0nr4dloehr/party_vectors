"""Cross-model logit-probe comparison (held-out selection split only).

The train split was used to compute the steering vectors, so only the
selection split is a proper unseen evaluation; train-split panels were
removed to avoid reporting in-split numbers.

Panel 1: per (model, party), the best-over-layers mean congruent-margin shift
at alpha=8 on the selection split — how steerable is each party in each
model, and at which layer.
Panel 2: forced-choice success rate on the selection split at the
train-selected best layer (layer chosen on train, evaluated on selection) —
the behavioral bottom line.

Usage: python viz/plot_cross_model_probe.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results_new" / "probe"
OUT = ROOT / "figures" / "logit_probe"
PARTIES = ["CDU_CSU", "GRUENE", "SPD", "AfD", "DIE_LINKE"]
MODELS = [
    "llama-3-2-1b-instruct",
    "llama-3-2-3b-instruct",
    "meta-llama-3-8b-instruct",
    "meta-llama-3-70b-instruct",
    "qwen3-8b",
    "ministral-8b-instruct-2410",
    "deepseek-llm-7b-chat",
    "gemma-4-e4b-it",
]
SHORT = {
    "llama-3-2-1b-instruct": "llama3-1b",
    "llama-3-2-3b-instruct": "llama3-3b",
    "meta-llama-3-8b-instruct": "llama3-8b",
    "meta-llama-3-70b-instruct": "llama3-70b",
    "qwen3-8b": "qwen3-8b",
    "ministral-8b-instruct-2410": "ministral-8b",
    "deepseek-llm-7b-chat": "deepseek-7b",
    "gemma-4-e4b-it": "gemma-4-e4b",
}


def load(model: str, party: str, split: str) -> pd.DataFrame:
    for name in [f"probe_results_{model}_{party}_{split}.csv",
                 f"probe_results_{model}_{party}.csv"]:
        path = RESULTS / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"{model}/{party}/{split}")


def best_layer_stats(model: str, party: str, split: str, alpha: float = 8.0):
    df = load(model, party, split)
    piv = df.pivot_table(index=["layer", "thesis_key"], columns="alpha", values="margin_cong")
    delta = (piv[alpha] - piv[0.0]).groupby("layer").mean()
    best_layer = int(delta.idxmax())
    # success rate at that layer
    d = df[(df["layer"] == best_layer)]
    sr_alpha = float(d[d["alpha"] == alpha]["argmax3_is_cong"].mean())
    sr_base = float(d[d["alpha"] == 0.0]["argmax3_is_cong"].mean())
    return best_layer, float(delta.max()), sr_alpha, sr_base


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    split = "selection"
    mat = pd.DataFrame(index=[SHORT[m] for m in MODELS], columns=PARTIES, dtype=float)
    layers = pd.DataFrame(index=[SHORT[m] for m in MODELS], columns=PARTIES, dtype=object)
    for m in MODELS:
        for p in PARTIES:
            layer, val, _, _ = best_layer_stats(m, p, split)
            mat.loc[SHORT[m], p] = val
            layers.loc[SHORT[m], p] = layer
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(mat.astype(float), annot=True, fmt=".1f", cmap="Reds", ax=ax,
                cbar_kws={"label": "best mean margin shift (alpha=8)"})
    for i, m in enumerate(mat.index):
        for j, p in enumerate(PARTIES):
            ax.text(j + 0.5, i + 0.82, f"L{layers.loc[m, p]}", ha="center",
                    va="center", fontsize=7, color="dimgray")
    ax.set_title("Best-layer congruent margin shift by model x party (selection split)",
                 fontweight="bold")
    fig.tight_layout()
    path = OUT / "cross_model_best_margin_selection.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # success-rate panel: layer selected on train, evaluated on selection
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    x = np.arange(len(MODELS) * len(PARTIES))
    sr_a, sr_b = [], []
    for m in MODELS:
        for p in PARTIES:
            layer, _, _, _ = best_layer_stats(m, p, "train")
            df = load(m, p, "selection")
            d = df[df["layer"] == layer]
            sr_a.append(float(d[d["alpha"] == 8.0]["argmax3_is_cong"].mean()))
            sr_b.append(float(d[d["alpha"] == 0.0]["argmax3_is_cong"].mean()))
    ax.bar(x - width / 2, sr_b, width, label="alpha=0 (baseline)", color="gray", alpha=0.7)
    ax.bar(x + width / 2, sr_a, width, label="alpha=8 @ train-best layer", color="firebrick")
    labels = [f"{SHORT[m]}\n{p}" for m in MODELS for p in PARTIES]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylabel("forced-choice success rate")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Selection-split success rate at the train-selected best layer "
                 "(alpha=8) vs baseline", fontweight="bold")
    fig.tight_layout()
    path = OUT / "cross_model_success_rates.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
