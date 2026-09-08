"""Shared data loading for ideology vector experiment visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

PARTIES: Dict[str, str] = {
    "CDU_CSU": "CDU / CSU",
    "GRUENE": "GRÜNE",
    "SPD": "SPD",
    "AfD": "AfD",
    "DIE_LINKE": "DIE LINKE",
}

MODELS = {
    "llama-3-2-1b-instruct": "Llama 3.2 1B",
    "llama-3-2-3b-instruct": "Llama 3.2 3B",
    "meta-llama-3-8b-instruct": "Llama 3 8B",
    "meta-llama-3-70b-instruct": "Llama 3 70B",
    "qwen3-8b": "Qwen3 8B",
    "ministral-8b-instruct-2410": "Ministral 8B",
    "deepseek-llm-7b-chat": "DeepSeek 7B",
    "gemma-4-e4b-it": "Gemma 4 E4B",
}

MODEL_ORDER = list(MODELS.keys())
PARTY_ORDER = list(PARTIES.keys())
SCHEMA_VERSION = 3
ALPHA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def model_label(model: str) -> str:
    return MODELS.get(model, model)


def parse_model_party_filename(name: str, prefix: str, suffix: str = ".csv") -> Optional[tuple]:
    """Parse ``{prefix}{model}_{party}{suffix}`` using known party keys."""
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    stem = name[len(prefix) : -len(suffix)]
    for party in PARTY_ORDER:
        token = f"_{party}"
        if stem.endswith(token):
            return stem[: -len(token)], party
    return None


def _ordered_models(found: Iterable[str]) -> List[str]:
    present = set(found)
    ordered = [model for model in MODEL_ORDER if model in present]
    extras = sorted(present.difference(MODEL_ORDER))
    return ordered + extras


def sweep_summary_path(model: str, steered_party: str) -> Path:
    return data_dir() / "steering" / f"sweep_summary_{model}_{steered_party}.csv"


def discover_steering_models() -> List[str]:
    """Return models that have at least one Phase-3 sweep summary on disk."""
    found = []
    steering_dir = data_dir() / "steering"
    if not steering_dir.exists():
        return []
    for path in steering_dir.glob("sweep_summary_*.csv"):
        parsed = parse_model_party_filename(path.name, "sweep_summary_")
        if parsed is not None:
            found.append(parsed[0])
    return _ordered_models(found)


def available_models(*, require_steering: bool = False) -> List[str]:
    """Return model keys that have result files on disk."""
    if require_steering:
        return discover_steering_models()
    found = []
    for path in data_dir().glob("baseline_results_*.csv"):
        found.append(path.name[len("baseline_results_") : -4])
    known = [model for model in _ordered_models(found)]
    if known:
        return known
    return [model for model in MODEL_ORDER if (data_dir() / f"baseline_results_{model}.csv").exists()]


def available_steering_parties(model: str) -> List[str]:
    return [party for party in PARTY_ORDER if sweep_summary_path(model, party).exists()]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    return project_root() / "data" / "results_new"


def output_dir() -> Path:
    out = project_root() / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _require_columns(df: pd.DataFrame, cols: List[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} missing columns {missing}. "
            "Regenerate results with schema_version=3 pipeline."
        )


def load_baseline(model: str) -> pd.DataFrame:
    path = data_dir() / f"baseline_results_{model}.csv"
    df = pd.read_csv(path)
    if "schema_version" in df.columns:
        _require_columns(df, ["thesis_key", "party_statement_key", "similarity_score"], path)
    return df


def load_neutral_baseline(model: str) -> pd.DataFrame:
    path = data_dir() / f"neutral_baseline_{model}_crossparty.csv"
    df = pd.read_csv(path)
    if "schema_version" in df.columns:
        _require_columns(df, ["thesis_key"], path)
    return df


def load_alpha0_baseline(model: str, steered_party: str) -> Optional[pd.DataFrame]:
    path = data_dir() / "steering" / f"alpha0_baseline_{model}_{steered_party}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_selected_config(model: str, steered_party: str) -> Optional[dict]:
    path = data_dir() / "steering" / f"selected_config_{model}_{steered_party}.json"
    if not path.exists():
        return None
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_sweep_summary(model: str, steered_party: str) -> pd.DataFrame:
    path = sweep_summary_path(model, steered_party)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "mean_delta" not in df.columns and "delta_vs_alpha0" in df.columns:
        df["mean_delta"] = df["delta_vs_alpha0"]
    df["model"] = model
    df["model_label"] = model_label(model)
    df["steered_party_key"] = steered_party
    df["steered_party"] = PARTIES[steered_party]
    df["compared_party"] = df["compared_party_key"].map(PARTIES)
    df["is_target_party"] = df["compared_party_key"] == steered_party
    return df


def load_all_summaries(models: Optional[Iterable[str]] = None) -> pd.DataFrame:
    models = list(models or available_models(require_steering=True))
    frames = [
        load_sweep_summary(model, party)
        for model in models
        for party in available_steering_parties(model)
    ]
    if not frames:
        raise FileNotFoundError(
            "No sweep summary CSV files found under data/results_new/steering/."
        )
    return pd.concat(frames, ignore_index=True)


def party_prompted_means(model: str) -> pd.Series:
    baseline = load_baseline(model)
    return baseline.groupby("party_key")["similarity_score"].mean().rename("party_prompted_mean")


def neutral_means(model: str) -> pd.Series:
    path = data_dir() / f"neutral_baseline_{model}_crossparty.csv"
    if not path.exists():
        return pd.Series(dtype=float, name="neutral_mean")
    neutral = load_neutral_baseline(model)
    return pd.Series(
        {pk: neutral[f"sim_{pk}"].mean() for pk in PARTY_ORDER if f"sim_{pk}" in neutral.columns},
        name="neutral_mean",
    )


def alpha0_means(model: str, steered_party: str) -> pd.Series:
    alpha0 = load_alpha0_baseline(model, steered_party)
    if alpha0 is not None:
        return pd.Series(
            {
                pk: alpha0[f"sim_{pk}"].mean()
                for pk in PARTY_ORDER
                if f"sim_{pk}" in alpha0.columns
            },
            name="alpha0_mean",
        )
    summary = load_sweep_summary(model, steered_party)
    sub = summary[(summary["alpha"] == 0.0) & (summary["split"] == "selection")]
    if sub.empty:
        sub = summary[summary["alpha"] == 0.0]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("compared_party_key")["mean_score"].mean()


def best_steering_config(summary: pd.DataFrame, steered_party: str, split: str = "selection") -> pd.DataFrame:
    sub = summary[
        (summary["steered_party_key"] == steered_party)
        & (summary["compared_party_key"] == steered_party)
        & (summary["split"] == split)
    ].copy()
    if sub.empty:
        sub = summary[
            (summary["steered_party_key"] == steered_party)
            & (summary["compared_party_key"] == steered_party)
        ].copy()
    if sub.empty or sub["mean_delta"].notna().sum() == 0:
        return pd.DataFrame()
    best = sub.loc[sub["mean_delta"].idxmax()]
    return pd.DataFrame([best])


def build_comparison_table(model: str) -> pd.DataFrame:
    party_prompted = party_prompted_means(model)
    neutral = neutral_means(model)
    summaries = load_all_summaries([model])
    rows = []
    for steered in PARTY_ORDER:
        sub = summaries[summaries["steered_party_key"] == steered]
        alpha0 = alpha0_means(model, steered)
        selected = load_selected_config(model, steered)
        best = best_steering_config(sub, steered, split="selection")
        if selected and selected.get("layer") is not None and selected.get("alpha") is not None:
            sel_layer = int(selected["layer"])
            sel_alpha = float(selected["alpha"])
            sel_delta = selected.get("selection_mean_delta")
            sel_rows = sub[
                (sub["compared_party_key"] == steered)
                & (sub["layer"] == sel_layer)
                & (sub["alpha"] == sel_alpha)
                & (sub["split"] == "selection")
            ]
            if sel_rows.empty:
                sel_rows = sub[
                    (sub["compared_party_key"] == steered)
                    & (sub["layer"] == sel_layer)
                    & (sub["alpha"] == sel_alpha)
                ]
            if not sel_rows.empty:
                sel_row = sel_rows.iloc[0]
                best_score = sel_row["mean_score"]
                best_delta = sel_row.get("mean_delta", sel_delta)
                best_layer = sel_layer
                best_alpha = sel_alpha
                best_ci_low = sel_row.get("ci_low", selected.get("selection_ci_low"))
                best_ci_high = sel_row.get("ci_high", selected.get("selection_ci_high"))
            elif not best.empty:
                best_row = best.iloc[0]
                best_score = best_row["mean_score"]
                best_delta = best_row["mean_delta"]
                best_layer = int(best_row["layer"])
                best_alpha = float(best_row["alpha"])
                best_ci_low = best_row.get("ci_low")
                best_ci_high = best_row.get("ci_high")
            else:
                best_score = best_delta = best_ci_low = best_ci_high = None
                best_layer = sel_layer
                best_alpha = sel_alpha
                best_delta = sel_delta
        elif not best.empty:
            best_row = best.iloc[0]
            best_score = best_row["mean_score"]
            best_delta = best_row["mean_delta"]
            best_layer = int(best_row["layer"])
            best_alpha = float(best_row["alpha"])
            best_ci_low = best_row.get("ci_low")
            best_ci_high = best_row.get("ci_high")
        else:
            best_score = best_delta = best_layer = best_alpha = best_ci_low = best_ci_high = None

        for pk in PARTY_ORDER:
            rows.append(
                {
                    "model": model,
                    "model_label": model_label(model),
                    "steered_party_key": steered,
                    "steered_party": PARTIES[steered],
                    "compared_party_key": pk,
                    "compared_party": PARTIES[pk],
                    "party_prompted_baseline": party_prompted.get(pk),
                    "neutral_baseline": neutral.get(pk),
                    "steering_alpha0": alpha0.get(pk),
                    "best_steered_score": best_score if pk == steered else None,
                    "best_steered_delta": best_delta if pk == steered else None,
                    "best_steered_ci_low": best_ci_low if pk == steered else None,
                    "best_steered_ci_high": best_ci_high if pk == steered else None,
                    "best_layer": best_layer if pk == steered else None,
                    "best_alpha": best_alpha if pk == steered else None,
                }
            )
    return pd.DataFrame(rows)


def cross_party_gain_matrix(model: str, split: str = "selection") -> pd.DataFrame:
    summaries = load_all_summaries([model])
    mat = pd.DataFrame(index=PARTY_ORDER, columns=PARTY_ORDER, dtype=float)
    for steered in PARTY_ORDER:
        sub = summaries[
            (summaries["steered_party_key"] == steered) & (summaries["split"] == split)
        ]
        for pk in PARTY_ORDER:
            party_rows = sub[sub["compared_party_key"] == pk]
            mat.loc[steered, pk] = (
                float(party_rows["mean_delta"].max()) if not party_rows.empty else float("nan")
            )
    mat.index = [PARTIES[k] for k in PARTY_ORDER]
    mat.columns = [PARTIES[k] for k in PARTY_ORDER]
    return mat


def load_sweep_results(model: str, steered_party: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    path = data_dir() / "steering" / f"sweep_results_{model}_{steered_party}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, usecols=columns)


def _paired_thesis_deltas(
    sweep: pd.DataFrame,
    party: str,
    layer: int,
    alpha: float,
) -> pd.Series:
    """Paired per-thesis Δ vs the same-layer α=0 row. Drops incomplete pairs."""
    col = f"sim_{party}"
    steered = sweep[(sweep["layer"] == layer) & (sweep["alpha"] == alpha)][["thesis_key", col]]
    baseline = sweep[(sweep["layer"] == layer) & (sweep["alpha"] == 0.0)][["thesis_key", col]]
    merged = steered.merge(baseline, on="thesis_key", suffixes=("_steered", "_alpha0"))
    merged = merged.dropna()
    return merged[f"{col}_steered"] - merged[f"{col}_alpha0"]


def load_selected_success_table(*, include_win_rates: bool = True) -> pd.DataFrame:
    """One row per (model, party) at the Phase-3 selected (layer, alpha).

    ``mean_delta`` / CIs are the official paired-summary metric. Win/tie/lose
    rates are computed on the same paired theses. Llama-3-70B rows can have
    ``n_pairs`` well below 57; callers should surface that.
    """
    rows = []
    for model in available_models(require_steering=True):
        for party in available_steering_parties(model):
            summary = load_sweep_summary(model, party)
            selected = load_selected_config(model, party)
            target = summary[
                (summary["compared_party_key"] == party) & (summary["split"] == "selection")
            ]
            if target.empty:
                target = summary[summary["compared_party_key"] == party]
            if target.empty or target["mean_delta"].notna().sum() == 0:
                continue
            if selected and selected.get("layer") is not None and selected.get("alpha") is not None:
                layer = int(selected["layer"])
                alpha = float(selected["alpha"])
            else:
                best = target.loc[target["mean_delta"].idxmax()]
                layer = int(best["layer"])
                alpha = float(best["alpha"])
            cell = target[(target["layer"] == layer) & (target["alpha"] == alpha)]
            if cell.empty:
                cell = pd.DataFrame([target.loc[target["mean_delta"].idxmax()]])
                layer = int(cell.iloc[0]["layer"])
                alpha = float(cell.iloc[0]["alpha"])
            row = cell.iloc[0]
            others = summary[
                (summary["layer"] == layer)
                & (summary["alpha"] == alpha)
                & (summary["compared_party_key"] != party)
            ]
            other_mean = float(others["mean_delta"].mean()) if not others.empty else float("nan")
            ci_low = float(row["ci_low"]) if pd.notna(row.get("ci_low")) else float("nan")
            ci_high = float(row["ci_high"]) if pd.notna(row.get("ci_high")) else float("nan")
            record = {
                "model": model,
                "model_label": model_label(model),
                "steered_party_key": party,
                "steered_party": PARTIES[party],
                "layer": layer,
                "alpha": alpha,
                "mean_score": float(row["mean_score"]) if pd.notna(row.get("mean_score")) else float("nan"),
                "mean_delta": float(row["mean_delta"]),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_pairs": int(row["n_pairs"]) if pd.notna(row.get("n_pairs")) else 0,
                "n_valid": int(row["n_valid"]) if pd.notna(row.get("n_valid")) else 0,
                "n_total": int(row["n_total"]) if pd.notna(row.get("n_total")) else 0,
                "other_party_mean_delta": other_mean,
                "specificity": float(row["mean_delta"]) - other_mean if pd.notna(other_mean) else float("nan"),
                "ci_excludes_zero": bool(pd.notna(ci_low) and ci_low > 0),
                "steers": bool(alpha > 0 and float(row["mean_delta"]) > 0),
                "incomplete_pairs": bool(int(row["n_pairs"]) < 57) if pd.notna(row.get("n_pairs")) else False,
            }
            if include_win_rates:
                sim_col = f"sim_{party}"
                try:
                    sweep = load_sweep_results(
                        model,
                        party,
                        columns=["thesis_key", "layer", "alpha", sim_col],
                    )
                    deltas = _paired_thesis_deltas(sweep, party, layer, alpha)
                    n = int(len(deltas))
                    record["win_rate"] = float((deltas > 0).mean()) if n else float("nan")
                    record["tie_rate"] = float((deltas == 0).mean()) if n else float("nan")
                    record["lose_rate"] = float((deltas < 0).mean()) if n else float("nan")
                    record["n_paired_theses"] = n
                except FileNotFoundError:
                    record["win_rate"] = float("nan")
                    record["tie_rate"] = float("nan")
                    record["lose_rate"] = float("nan")
                    record["n_paired_theses"] = 0
            rows.append(record)
    if not rows:
        raise FileNotFoundError("No selected steering configs found under data/results_new/steering/.")
    return pd.DataFrame(rows)


def load_fixed_alpha_success_table() -> pd.DataFrame:
    """One row per (model, party, alpha), holding injection strength fixed.

    At each alpha the layer with the largest target-party ``mean_delta`` is
    kept. Parties and models can then be compared at the same α; layer is
    allowed to differ because the targeted sweep uses a different probe-chosen
    trio per party.
    """
    rows = []
    for model in available_models(require_steering=True):
        for party in available_steering_parties(model):
            summary = load_sweep_summary(model, party)
            target = summary[summary["compared_party_key"] == party]
            if "split" in target.columns:
                selection = target[target["split"] == "selection"]
                if not selection.empty:
                    target = selection
            for alpha, group in target.groupby("alpha"):
                if group["mean_delta"].notna().sum() == 0:
                    continue
                best = group.loc[group["mean_delta"].idxmax()]
                layer = int(best["layer"])
                others = summary[
                    (summary["layer"] == layer)
                    & (summary["alpha"] == float(alpha))
                    & (summary["compared_party_key"] != party)
                ]
                other_mean = float(others["mean_delta"].mean()) if not others.empty else float("nan")
                ci_low = float(best["ci_low"]) if pd.notna(best.get("ci_low")) else float("nan")
                ci_high = float(best["ci_high"]) if pd.notna(best.get("ci_high")) else float("nan")
                n_pairs = int(best["n_pairs"]) if pd.notna(best.get("n_pairs")) else 0
                rows.append(
                    {
                        "model": model,
                        "model_label": model_label(model),
                        "steered_party_key": party,
                        "steered_party": PARTIES[party],
                        "alpha": float(alpha),
                        "layer": layer,
                        "mean_score": float(best["mean_score"]) if pd.notna(best.get("mean_score")) else float("nan"),
                        "mean_delta": float(best["mean_delta"]),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_pairs": n_pairs,
                        "n_valid": int(best["n_valid"]) if pd.notna(best.get("n_valid")) else 0,
                        "other_party_mean_delta": other_mean,
                        "specificity": float(best["mean_delta"]) - other_mean if pd.notna(other_mean) else float("nan"),
                        "ci_excludes_zero": bool(pd.notna(ci_low) and ci_low > 0),
                        "incomplete_pairs": n_pairs < 57,
                    }
                )
    if not rows:
        raise FileNotFoundError("No sweep summaries found under data/results_new/steering/.")
    return pd.DataFrame(rows)


def load_cross_party_fixed_alpha_table() -> pd.DataFrame:
    """One row per (model, steered party, judged party, alpha).

    Injection strength is held fixed. At each α the layer is the one with the
    largest *target-party* mean Δ among the three probe-selected layers, then
    all five judged-party scores at that (layer, α) are emitted. That is the
    cross-party profile at a comparable α.
    """
    rows = []
    for model in available_models(require_steering=True):
        for steered in available_steering_parties(model):
            summary = load_sweep_summary(model, steered)
            if "split" in summary.columns:
                selection = summary[summary["split"] == "selection"]
                if not selection.empty:
                    summary = selection
            target = summary[summary["compared_party_key"] == steered]
            for alpha, group in target.groupby("alpha"):
                if group["mean_delta"].notna().sum() == 0:
                    continue
                best = group.loc[group["mean_delta"].idxmax()]
                layer = int(best["layer"])
                slice_rows = summary[
                    (summary["layer"] == layer) & (summary["alpha"] == float(alpha))
                ]
                baseline_rows = summary[
                    (summary["layer"] == layer) & (summary["alpha"] == 0.0)
                ]
                baseline_score = {
                    str(r["compared_party_key"]): float(r["mean_score"])
                    for _, r in baseline_rows.iterrows()
                    if pd.notna(r.get("mean_score"))
                }
                for _, row in slice_rows.iterrows():
                    judged = str(row["compared_party_key"])
                    n_pairs = int(row["n_pairs"]) if pd.notna(row.get("n_pairs")) else 0
                    rows.append(
                        {
                            "model": model,
                            "model_label": model_label(model),
                            "steered_party_key": steered,
                            "steered_party": PARTIES[steered],
                            "compared_party_key": judged,
                            "compared_party": PARTIES.get(judged, judged),
                            "is_target_party": judged == steered,
                            "alpha": float(alpha),
                            "layer": layer,
                            "mean_score": float(row["mean_score"]) if pd.notna(row.get("mean_score")) else float("nan"),
                            "alpha0_score": baseline_score.get(judged, float("nan")),
                            "mean_delta": float(row["mean_delta"]) if pd.notna(row.get("mean_delta")) else float("nan"),
                            "ci_low": float(row["ci_low"]) if pd.notna(row.get("ci_low")) else float("nan"),
                            "ci_high": float(row["ci_high"]) if pd.notna(row.get("ci_high")) else float("nan"),
                            "n_pairs": n_pairs,
                            "incomplete_pairs": n_pairs < 57,
                        }
                    )
    if not rows:
        raise FileNotFoundError("No sweep summaries found under data/results_new/steering/.")
    return pd.DataFrame(rows)
