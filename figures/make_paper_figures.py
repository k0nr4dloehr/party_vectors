"""Generate all paper figures from raw results data.

Single source of truth for publication figures referenced by paper/main.tex.
Reads probe, steering, vector, and annotation data from data/ and writes
vector PDFs (+ PNG previews) to figures/paper/.

All numbers are recomputed from raw per-thesis results; nothing is
hard-coded from the manuscript.

Figures
-------
fig_headline_grid        model x party grid: probe strength vs generation effect
fig_probe_heatmap_8b     layer x alpha probe heatmaps, Llama-3-8B, all parties
fig_generation_selected  strongest generation effect per model and party, CIs
fig_afd_dose_l31         dose-response, Llama-3-8B AfD direction at layer 31
fig_vector_geometry      party-vector cosines and norms by layer, Llama-3-8B
fig_headroom             unsteered margin vs best probe shift (headroom confound)
fig_crossparty_gain      steered-vs-judged similarity change matrix (5x5)
fig_judge_calibration    human vs LLM judge scores, annotated subset
fig_norms_depth          mean vector-norm profile by relative depth, 8 models
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "results_new"
VECTORS = ROOT / "data" / "vectors_new"
ANNOTATION = ROOT / "data" / "human_annotation"
OUT = ROOT / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

# Party keys as used in the data files
PARTY_KEYS = ["CDU_CSU", "SPD", "GRUENE", "AfD", "DIE_LINKE"]
PARTY_LABELS = {
    "CDU_CSU": "CDU/CSU",
    "SPD": "SPD",
    "GRUENE": "GR\u00dcNE",
    "AfD": "AfD",
    "DIE_LINKE": "Die Linke",
}
# Display order: left to right of the German political spectrum
PARTY_ORDER = ["DIE_LINKE", "SPD", "GRUENE", "CDU_CSU", "AfD"]
# Conventional German party colors
PARTY_COLOR = {
    "CDU_CSU": "#333333",
    "SPD": "#E3000F",
    "GRUENE": "#1AA737",
    "AfD": "#009EE0",
    "DIE_LINKE": "#B61B8E",
}

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
MODEL_LABELS = {
    "llama-3-2-1b-instruct": "Llama-3.2-1B",
    "llama-3-2-3b-instruct": "Llama-3.2-3B",
    "meta-llama-3-8b-instruct": "Llama-3-8B",
    "meta-llama-3-70b-instruct": "Llama-3-70B",
    "qwen3-8b": "Qwen3-8B",
    "ministral-8b-instruct-2410": "Ministral-8B",
    "deepseek-llm-7b-chat": "DeepSeek-7B",
    "gemma-4-e4b-it": "Gemma-4 E4B",
}
MODEL_COLOR = {
    "llama-3-2-1b-instruct": "#7f9ec9",
    "llama-3-2-3b-instruct": "#4a6fb0",
    "meta-llama-3-8b-instruct": "#1f4e79",
    "meta-llama-3-70b-instruct": "#0d2f4f",
    "qwen3-8b": "#8e44ad",
    "ministral-8b-instruct-2410": "#e67e22",
    "deepseek-llm-7b-chat": "#16a085",
    "gemma-4-e4b-it": "#c0392b",
}

CMAP_DIVERGING = "RdBu"  # red = up, blue = down (colorblind-safe pairing)

# Consistent style across all figures
plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_selected_configs() -> dict[tuple[str, str], dict]:
    """Load the selected (layer, alpha) configuration per model and party.

    The party suffix is matched against the known party keys; a naive
    rsplit on '_' would break CDU_CSU and DIE_LINKE filenames.
    """
    configs = {}
    for f in sorted((DATA / "steering").glob("selected_config_*.json")):
        stem = f.stem.replace("selected_config_", "")
        party = next((p for p in PARTY_KEYS if stem.endswith(f"_{p}")), None)
        if party is None:
            raise ValueError(f"cannot parse party from {f.name}")
        model = stem[: -len(party) - 1]
        with open(f, encoding="utf-8") as fh:
            configs[(model, party)] = json.load(fh)
    return configs


def load_sweep_summary(model: str, party: str) -> pd.DataFrame:
    return pd.read_csv(DATA / "steering" / f"sweep_summary_{model}_{party}.csv")


def load_probe(model: str, party: str, split: str | None = None) -> pd.DataFrame:
    """Load probe results; `split` in {'train', 'selection'} if given.

    File naming is inconsistent across models: most models use
    probe_results_<model>_<party>_<split>.csv, while the 8B files carry
    the selection split in probe_results_<model>_<party>.csv without a
    suffix (the train split always carries the _train suffix).
    """
    candidates = []
    if split is not None:
        candidates.append(DATA / "probe" / f"probe_results_{model}_{party}_{split}.csv")
    candidates.append(DATA / "probe" / f"probe_results_{model}_{party}.csv")
    for f in candidates:
        if f.exists():
            df = pd.read_csv(f)
            if split is not None and "split" in df.columns:
                df = df[df["split"] == split]
            return df
    raise FileNotFoundError(f"no probe file for {model} / {party} / {split}")


def probe_margin_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """Mean congruent-margin shift vs same-layer alpha=0, per (layer, alpha).

    Returns a DataFrame indexed by (layer, alpha) with columns
    'delta_margin' (mean over theses) and 'n'.
    """
    base = df[df["alpha"] == 0].groupby("layer")["margin_cong"].mean()
    rows = []
    for (layer, alpha), g in df.groupby(["layer", "alpha"]):
        if alpha == 0:
            continue
        if layer not in base.index:
            continue
        rows.append(
            {
                "layer": layer,
                "alpha": alpha,
                "delta_margin": g["margin_cong"].mean() - base.loc[layer],
                "n": len(g),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["layer", "alpha", "delta_margin", "n"])
    return pd.DataFrame(rows)


def probe_best_margin(df: pd.DataFrame) -> float:
    """Best mean congruent-margin shift vs alpha=0 across layers/alphas."""
    shifts = probe_margin_shifts(df)
    if shifts.empty:
        return np.nan
    return float(shifts["delta_margin"].max())


def load_party_vectors(model: str, party: str) -> dict[int, torch.Tensor]:
    """Load the stored (unnormalized) steering vectors for one party."""
    base = VECTORS / model / party
    meta = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
    n = meta["num_layers"]
    return {
        layer: torch.load(
            base / f"vector_layer_{layer:02d}.pt", map_location="cpu", weights_only=True
        )
        for layer in range(n)
    }


def save(fig: plt.Figure, name: str) -> None:
    """Write a figure as vector PDF plus a PNG preview."""
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"wrote {name}.pdf")


# ---------------------------------------------------------------------------
# Figure: headline grid (probe vs generation, model x party)
# ---------------------------------------------------------------------------


def fig_headline_grid(configs: dict) -> plt.Figure:
    """Model x party grid showing probe strength and generation effect.

    Panel (a): best mean congruent-margin shift found anywhere in the
    probe sweep. Panel (b): generation delta at the strongest tested
    steering result, outlined where its 95% interval excludes zero.
    Together they show the transfer gap between the two instruments.
    """
    n_models = len(MODELS)
    n_parties = len(PARTY_ORDER)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.4),
        gridspec_kw={"width_ratios": [1, 1.4]},
    )

    # -- Panel (a): probe strength (best delta-margin) -------------------
    ax = axes[0]
    probe_vals = np.full((n_models, n_parties), np.nan)
    for i, model in enumerate(MODELS):
        for j, party in enumerate(PARTY_ORDER):
            try:
                df = load_probe(model, party, split="selection")
                if df.empty:
                    df = load_probe(model, party, split="train")
            except FileNotFoundError:
                continue
            if not df.empty:
                probe_vals[i, j] = probe_best_margin(df)

    vmax = np.nanmax(np.abs(probe_vals))
    im = ax.imshow(
        probe_vals,
        cmap=CMAP_DIVERGING,
        norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
        aspect="auto",
    )
    ax.set_xticks(range(n_parties))
    ax.set_xticklabels([PARTY_LABELS[p] for p in PARTY_ORDER], rotation=45, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_title("(a) Probe: best layer, forced choice", loc="left", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("congruent-margin shift (logits)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    for i in range(n_models):
        for j in range(n_parties):
            v = probe_vals[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j,
                i,
                f"{v:+.1f}",
                ha="center",
                va="center",
                fontsize=6,
                color="black" if abs(v) < vmax * 0.55 else "white",
            )

    # -- Panel (b): generation delta at the strongest tested result ------
    ax = axes[1]
    gen_vals = np.full((n_models, n_parties), np.nan)
    ci_excl = np.zeros((n_models, n_parties), dtype=bool)
    for i, model in enumerate(MODELS):
        for j, party in enumerate(PARTY_ORDER):
            cfg = configs.get((model, party))
            if cfg is None:
                continue
            gen_vals[i, j] = cfg["selection_mean_delta"]
            if cfg["selection_ci_low"] > 0:
                ci_excl[i, j] = True

    vmax = np.nanmax(np.abs(gen_vals)) * 1.05
    im = ax.imshow(
        gen_vals,
        cmap=CMAP_DIVERGING,
        norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
        aspect="auto",
    )
    ax.set_xticks(range(n_parties))
    ax.set_xticklabels([PARTY_LABELS[p] for p in PARTY_ORDER], rotation=45, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([])
    ax.set_title("(b) Generation: strongest tested result", loc="left", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("judge-score change (1-10)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    for i in range(n_models):
        for j in range(n_parties):
            v = gen_vals[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j,
                i,
                f"{v:+.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="black" if abs(v) < vmax * 0.55 else "white",
            )
    # bold outline on cells whose CI excludes zero
    for i in range(n_models):
        for j in range(n_parties):
            if ci_excl[i, j]:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        lw=1.6,
                        zorder=10,
                    )
                )

    fig.tight_layout(w_pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# Figure: probe heatmap 8B (layer x alpha per party)
# ---------------------------------------------------------------------------


def fig_probe_heatmap_8b() -> plt.Figure:
    """Layer x alpha heatmaps of congruent-margin shift for Llama-3-8B.

    One panel per party; shows where (layer, strength) the extracted
    direction moves the forced-choice stance, including sign flips.
    """
    model = "meta-llama-3-8b-instruct"
    alphas = [0.5, 1, 2, 4, 8]
    n_layers = 32
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 2.9), sharey=True)
    vmax = 0.0
    data = {}
    for party in PARTY_ORDER:
        df = load_probe(model, party, split="selection")
        base = df[df["alpha"] == 0].groupby("layer")["margin_cong"].mean()
        grid = np.full((n_layers, len(alphas)), np.nan)
        for (layer, alpha), g in df.groupby(["layer", "alpha"]):
            if alpha not in alphas or layer >= n_layers:
                continue
            grid[layer, alphas.index(alpha)] = g["margin_cong"].mean() - base.loc[layer]
        data[party] = grid
        vmax = max(vmax, np.nanmax(np.abs(grid)))
    for k, party in enumerate(PARTY_ORDER):
        ax = axes[k]
        im = ax.imshow(
            data[party],
            cmap=CMAP_DIVERGING,
            norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
            aspect="auto",
            origin="lower",
        )
        ax.set_title(PARTY_LABELS[party], fontsize=8)
        ax.set_xlabel("strength $\\alpha$")
        if k == 0:
            ax.set_ylabel("layer")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels(alphas)
        ax.set_yticks([0, 8, 16, 24, 31])
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label("congruent-margin shift (logits)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    return fig


# ---------------------------------------------------------------------------
# Figure: strongest generation effect per model and party, with CIs
# ---------------------------------------------------------------------------


def fig_generation_selected(configs: dict) -> plt.Figure:
    """Strongest tested generation effect per model and party, with 95% CIs.

    Horizontal bar chart, one row per model, grouped by party color;
    bars whose CI excludes zero are filled, others hollow.
    """
    fig, ax = plt.subplots(figsize=(3.4, 4.6))
    y = 0
    yticks = []
    yticklabels = []
    for model in MODELS:
        for party in PARTY_ORDER:
            cfg = configs.get((model, party))
            if cfg is None:
                continue
            delta = cfg["selection_mean_delta"]
            lo = cfg["selection_ci_low"]
            hi = cfg["selection_ci_high"]
            sig = lo > 0
            ax.barh(
                y,
                delta,
                height=0.62,
                color=PARTY_COLOR[party] if sig else "none",
                edgecolor=PARTY_COLOR[party],
                lw=0.9,
            )
            ax.plot([lo, hi], [y, y], color=PARTY_COLOR[party], lw=0.8)
            y += 1
        yticks.append(y - 2.5)
        yticklabels.append(MODEL_LABELS[model])
        y += 1  # gap between models
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.invert_yaxis()
    ax.set_xlabel("judge-score change at the strongest tested result (1-10 scale)")
    ax.set_title("Open-ended generation", loc="left")
    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#555",
            markersize=7,
            label="95% interval excludes 0",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="white",
            markeredgecolor="#555",
            markersize=7,
            label="interval includes 0",
        ),
    ]
    ax.legend(handles=legend_elems, loc="lower right", frameon=False)
    return fig


# ---------------------------------------------------------------------------
# Figure: dose-response, Llama-3-8B AfD direction at layer 31
# ---------------------------------------------------------------------------


def fig_afd_dose_l31() -> plt.Figure:
    """Dose-response: judge-score change vs strength, 8B AfD, layer 31.

    The AfD score rises at moderate strengths and collapses at strength 8;
    the mean change of the other four parties moves the other way,
    showing the effect is party-specific rather than a generic shift.
    """
    model = "meta-llama-3-8b-instruct"
    party = "AfD"
    layer = 31
    sw = load_sweep_summary(model, party)
    sw = sw[sw["layer"] == layer]
    tgt = sw[sw["compared_party_key"] == party].sort_values("alpha")
    others = sw[sw["compared_party_key"] != party].groupby("alpha").agg(
        mean_delta=("mean_delta", "mean"), ci_low=("ci_low", "mean"), ci_high=("ci_high", "mean")
    )

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    (line,) = ax.plot(
        tgt["alpha"],
        tgt["mean_delta"],
        marker="o",
        ms=4,
        lw=1.4,
        color=PARTY_COLOR["AfD"],
        label="AfD (target)",
    )
    ax.fill_between(
        tgt["alpha"],
        tgt["ci_low"],
        tgt["ci_high"],
        alpha=0.18,
        color=line.get_color(),
    )
    (line2,) = ax.plot(
        others.index,
        others["mean_delta"],
        marker="s",
        ms=3.5,
        lw=1.2,
        color="#666666",
        label="other four parties (mean)",
    )
    ax.fill_between(
        others.index,
        others["ci_low"],
        others["ci_high"],
        alpha=0.15,
        color=line2.get_color(),
    )
    ax.axhline(0, color="black", lw=0.7, ls=":")
    ax.set_xscale("symlog", linthresh=0.25)
    ax.set_xticks([0, 0.25, 0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0", "0.25", "0.5", "1", "2", "4", "8"])
    ax.set_xlabel("steering strength $\\alpha$")
    ax.set_ylabel("judge-score change")
    ax.set_title("Llama-3-8B, AfD direction, layer 31", loc="left")
    ax.legend(frameon=False, loc="upper left")
    return fig


# ---------------------------------------------------------------------------
# Figure: vector geometry (Llama-3-8B), recomputed from stored vectors
# ---------------------------------------------------------------------------


def fig_vector_geometry() -> plt.Figure:
    """Party-vector geometry for Llama-3-8B, from the stored vectors.

    Left: mean pairwise cosine across the five party vectors per layer,
    with the min-max band and the -1/(n-1) reference line of a perfectly
    centered set. Right: per-pair cosine curves for the most opposed and
    most aligned party pairs, showing the pair structure beyond the mean.
    """
    model = "meta-llama-3-8b-instruct"
    vecs = {p: load_party_vectors(model, p) for p in PARTY_KEYS}
    n_layers = len(vecs[PARTY_KEYS[0]])

    cos_mean, cos_min, cos_max = [], [], []
    pair_cos: dict[tuple[str, str], list[float]] = {}
    for layer in range(n_layers):
        V = torch.stack([vecs[p][layer] for p in PARTY_KEYS]).float()
        V = V / V.norm(dim=1, keepdim=True).clamp(min=1e-12)
        C = (V @ V.T).numpy()
        off = C[~np.eye(len(PARTY_KEYS), dtype=bool)]
        cos_mean.append(off.mean())
        cos_min.append(off.min())
        cos_max.append(off.max())
        for i in range(len(PARTY_KEYS)):
            for j in range(i + 1, len(PARTY_KEYS)):
                pair_cos.setdefault((PARTY_KEYS[i], PARTY_KEYS[j]), []).append(
                    float(C[i, j])
                )

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    layers = np.arange(n_layers)

    ax = axes[0]
    ax.axhline(-0.25, color="0.45", lw=0.9, ls="--", label=r"$-1/(n-1)=-0.25$")
    ax.plot(layers, cos_mean, color="#1f4e79", lw=1.6)
    ax.fill_between(
        layers,
        cos_min,
        cos_max,
        color="#1f4e79",
        alpha=0.12,
        linewidth=0,
    )
    ax.set_ylim(-1.0, 0.75)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pairwise cosine")
    ax.set_title("(a) Mean pairwise cosine", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    ax = axes[1]
    # Most opposed and most aligned pairs, averaged over layers
    pair_means = {k: np.mean(v) for k, v in pair_cos.items()}
    most_opposed = min(pair_means, key=pair_means.get)
    most_aligned = max(pair_means, key=pair_means.get)
    ax.plot(
        layers,
        pair_cos[most_opposed],
        lw=1.4,
        color="#8c2d04",
        label=f"{PARTY_LABELS[most_opposed[0]]} \u2013 {PARTY_LABELS[most_opposed[1]]}",
    )
    ax.plot(
        layers,
        pair_cos[most_aligned],
        lw=1.4,
        color="#1f4e79",
        label=f"{PARTY_LABELS[most_aligned[0]]} \u2013 {PARTY_LABELS[most_aligned[1]]}",
    )
    # ideological-opposites pair for reference: AfD - Die Linke
    af = pair_cos.get(("AfD", "DIE_LINKE"))
    if af is not None:
        ax.plot(
            layers,
            af,
            lw=1.2,
            ls="--",
            color="#666666",
            label="AfD \u2013 Die Linke",
        )
    ax.axhline(0, color="0.75", lw=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pairwise cosine")
    ax.set_title("(b) Most opposed vs. most aligned pair", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    return fig


# ---------------------------------------------------------------------------
# Figure: headroom (unsteered margin vs best probe shift)
# ---------------------------------------------------------------------------


def fig_headroom() -> plt.Figure:
    """Unsteered congruent margin vs best achievable probe shift.

    One point per model and party (selection split). Models that start
    far from a party's stance have the most room to move; the resulting
    correlation (r ~ -0.9) shows probe-measured steerability is largely
    headroom rather than representation strength.
    """
    rows = []
    for model in MODELS:
        for party in PARTY_ORDER:
            try:
                df = load_probe(model, party, split="selection")
                if df.empty:
                    df = load_probe(model, party, split="train")
            except FileNotFoundError:
                continue
            if df.empty:
                continue
            base = df[df["alpha"] == 0]["margin_cong"].mean()
            best = probe_best_margin(df)
            rows.append(
                {
                    "model": model,
                    "party": party,
                    "baseline_margin": base,
                    "best_shift": best,
                }
            )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    for party in PARTY_ORDER:
        sub = df[df["party"] == party]
        ax.scatter(
            sub["baseline_margin"],
            sub["best_shift"],
            s=18,
            color=PARTY_COLOR[party],
            label=PARTY_LABELS[party],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.3,
        )
    # Correlation across all model-party points
    r = np.corrcoef(df["baseline_margin"], df["best_shift"])[0, 1]
    # Linear fit for trend line
    coef = np.polyfit(df["baseline_margin"], df["best_shift"], 1)
    xs = np.linspace(df["baseline_margin"].min(), df["baseline_margin"].max(), 100)
    ax.plot(xs, np.polyval(coef, xs), color="0.3", lw=1.0, ls="--")
    ax.set_xlabel("unsteered congruent margin (logits, $\\alpha=0$)")
    ax.set_ylabel("best probe shift (logits)")
    ax.set_title(f"Headroom: baseline predicts shift ($r={r:.2f}$)", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6, loc="upper right", ncol=2)
    return fig


# ---------------------------------------------------------------------------
# Figure: cross-party gain matrix (steered vs judged)
# ---------------------------------------------------------------------------


def fig_crossparty_gain(configs: dict) -> plt.Figure:
    """Cross-party structure of generation steering.

    Left: 5x5 matrix of mean judge-score change (steered party rows,
    judged party columns), averaged over the strongest tested result of
    each model. Right: two exemplar matrices (Llama-3-70B CDU/CSU,
    Gemma-4 AfD) that show the left-right block structure and the
    anti-specific AfD direction in Gemma.
    """
    from itertools import product

    # -- Aggregate matrix over all models, at each model's selected config
    agg = np.full((5, 5), np.nan)
    for i, steered in enumerate(PARTY_KEYS):
        vals = []
        for model in MODELS:
            cfg = configs.get((model, steered))
            if cfg is None:
                continue
            layer, alpha = cfg["layer"], cfg["alpha"]
            sw = load_sweep_summary(model, steered)
            sub = sw[(sw["layer"] == layer) & (sw["alpha"] == alpha)]
            for j, judged in enumerate(PARTY_KEYS):
                v = sub[sub["compared_party_key"] == judged]["mean_delta"]
                if len(v):
                    if np.isnan(agg[i, j]):
                        agg[i, j] = 0.0
                    agg[i, j] += float(v.iloc[0])
                # accumulate sums; divide by model count below
        # count models for averaging
    # Recompute properly with counts
    sums = np.full((5, 5), np.nan)
    counts = np.zeros((5, 5))
    for model in MODELS:
        for i, steered in enumerate(PARTY_KEYS):
            cfg = configs.get((model, steered))
            if cfg is None:
                continue
            layer, alpha = cfg["layer"], cfg["alpha"]
            sw = load_sweep_summary(model, steered)
            sub = sw[(sw["layer"] == layer) & (sw["alpha"] == alpha)]
            for j, judged in enumerate(PARTY_KEYS):
                v = sub[sub["compared_party_key"] == judged]["mean_delta"]
                if len(v):
                    if np.isnan(sums[i, j]):
                        sums[i, j] = 0.0
                    sums[i, j] += float(v.iloc[0])
                    counts[i, j] += 1
    agg = sums / counts

    # -- Exemplar matrices
    exemplars = [
        ("meta-llama-3-70b-instruct", "CDU_CSU"),
        ("gemma-4-e4b-it", "AfD"),
    ]
    exemplar_mats = []
    for model, steered in exemplars:
        cfg = configs[(model, steered)]
        sw = load_sweep_summary(model, steered)
        sub = sw[(sw["layer"] == cfg["layer"]) & (sw["alpha"] == cfg["alpha"])]
        mat = np.full((5, 5), np.nan)
        for i, s in enumerate(PARTY_KEYS):
            for j, judged in enumerate(PARTY_KEYS):
                v = sub[sub["compared_party_key"] == judged]["mean_delta"]
                if len(v):
                    mat[i, j] = float(v.iloc[0])
        exemplar_mats.append((model, steered, mat))

    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1, 1]}
    )
    mats = [("Mean over models", agg)] + [
        (f"{MODEL_LABELS[m]}, {PARTY_LABELS[s]}", mat) for m, s, mat in exemplar_mats
    ]
    vmax = np.nanmax(np.abs([m for _, m in mats]))
    for ax, (title, mat) in zip(axes, mats):
        im = ax.imshow(
            mat,
            cmap=CMAP_DIVERGING,
            norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
            aspect="equal",
        )
        ax.set_xticks(range(5))
        ax.set_xticklabels([PARTY_LABELS[p] for p in PARTY_KEYS], rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(5))
        ax.set_yticklabels([PARTY_LABELS[p] for p in PARTY_KEYS], fontsize=6)
        ax.set_title(title, loc="left", fontsize=8)
        for i in range(5):
            for j in range(5):
                if not np.isnan(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black" if abs(mat[i, j]) < vmax * 0.55 else "white",
                    )
    axes[0].set_xlabel("judged party")
    axes[0].set_ylabel("steered party")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("judge-score change", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    return fig


# ---------------------------------------------------------------------------
# Figure: judge calibration (human vs LLM)
# ---------------------------------------------------------------------------


def fig_judge_calibration() -> plt.Figure:
    """Human vs LLM judge scores on the annotated items.

    Scatter with identity line, colored by condition. Annotated with
    Pearson r. Only items with a human score are shown.
    """
    gold = pd.read_csv(ANNOTATION / "gold.csv")
    ann = pd.read_excel(ANNOTATION / "annotators all.xlsx", sheet_name="Annotation")
    df = gold.merge(ann[["item_id", "human_score"]], on="item_id", how="inner")
    df = df.dropna(subset=["human_score", "llm_score"])

    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    cond_colors = {
        "steered": PARTY_COLOR["AfD"],
        "unsteered": "#666666",
        "party_prompted": "#1AA737",
    }
    cond_labels = {
        "steered": "steered",
        "unsteered": "unsteered",
        "party_prompted": "party-prompted",
    }
    for cond in ["steered", "unsteered", "party_prompted"]:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        ax.scatter(
            sub["llm_score"],
            sub["human_score"],
            s=16,
            alpha=0.7,
            color=cond_colors[cond],
            label=f"{cond_labels[cond]} (n={len(sub)})",
            edgecolors="white",
            linewidths=0.3,
        )
    ax.plot([1, 10], [1, 10], color="0.3", lw=0.9, ls="--", label="identity")
    r = np.corrcoef(df["human_score"], df["llm_score"])[0, 1]
    ax.set_xlabel("LLM judge score")
    ax.set_ylabel("human score")
    ax.set_title(f"Judge validation ($r={r:.2f}$)", loc="left", fontsize=8.5)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    return fig


# ---------------------------------------------------------------------------
# Figure: vector norms by relative depth, all models
# ---------------------------------------------------------------------------


def fig_norms_depth() -> plt.Figure:
    """Mean party-vector norm by relative depth, for all eight models.

    Each curve is the mean L2 norm of the five party vectors at a layer,
    scaled to its model's maximum, plotted against relative depth
    (layer / num_layers). Highlights the last-layer blow-up for most
    models and the peak-then-collapse of DeepSeek and Qwen3.
    """
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for model in MODELS:
        try:
            vecs = {p: load_party_vectors(model, p) for p in PARTY_KEYS}
        except (FileNotFoundError, KeyError):
            continue
        n_layers = len(vecs[PARTY_KEYS[0]])
        depths = []
        norms = []
        for layer in range(n_layers):
            norms.append(
                float(torch.stack([vecs[p][layer] for p in PARTY_KEYS]).float().norm(dim=1).mean())
            )
            depths.append(layer / (n_layers - 1))
        norms = np.array(norms)
        norms_scaled = norms / norms.max()
        ax.plot(
            depths,
            norms_scaled,
            lw=1.2,
            color=MODEL_COLOR[model],
            label=MODEL_LABELS[model],
        )
    ax.set_xlabel("relative depth (layer / num layers)")
    ax.set_ylabel("mean party-vector norm (scaled)")
    ax.set_title("Vector norms by relative depth", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6, loc="upper left", ncol=2)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    configs = load_selected_configs()

    fig = fig_headline_grid(configs)
    save(fig, "fig_headline_grid")

    fig = fig_probe_heatmap_8b()
    save(fig, "fig_probe_heatmap_8b")

    fig = fig_generation_selected(configs)
    save(fig, "fig_generation_selected")

    fig = fig_afd_dose_l31()
    save(fig, "fig_afd_dose_l31")

    fig = fig_vector_geometry()
    save(fig, "fig_vector_geometry")

    fig = fig_headroom()
    save(fig, "fig_headroom")

    fig = fig_crossparty_gain(configs)
    save(fig, "fig_crossparty_gain")

    fig = fig_judge_calibration()
    save(fig, "fig_judge_calibration")

    fig = fig_norms_depth()
    save(fig, "fig_norms_depth")


if __name__ == "__main__":
    main()