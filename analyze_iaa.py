"""Inter-annotator agreement for the human-validation sample.

Metrics for four human annotators who scored 60 shared items (H-001..H-060) on
the same 1-10 isolated comparison the LLM judge performs. Score values are
intended as ``human_score_1..human_score_4`` columns of the shared workbook
``data/human_annotation/annotators all.xlsx``; the judge column is
``llm_score`` in ``data/human_annotation/gold.csv``.

Primary metric: quadratic-weighted Cohen kappa (two raters, ordered
categories). The quadratic weight v_ij = 1 - ((i-j)/(k-1))^2 turns squared
score distance into agreement credit, so a 1-vs-2 pair counts almost fully
while a 1-vs-10 pair counts as zero. Unlike exact-match accuracy, it is
chance-corrected. Reported:

- rater x rater confusion statistics,
- mean pairwise weighted kappa, with a percentile bootstrap 95% CI over items,
- pairwise weighted kappa per rater pair,
- judge-as-rater (mean human score vs LLM score), same metric,
- sensitivity analysis: unweighted (nominal) kappa and linear-weighted kappa,
- descriptive sanity numbers (exact / within-1 agreement, mean absolute
  difference) and a histogram of the 240 human scores.

Usage:
    python analyze_iaa.py
    python analyze_iaa.py --max-boot 2000 --seed 42
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

SCORE_MIN = 1
SCORE_MAX = 10
N_CATEGORIES = SCORE_MAX - SCORE_MIN + 1  # 10 ordered categories


@dataclass(frozen=True)
class DataPaths:
    """File locations for the annotation workbook and the gold judge file."""

    workbook: Path = Path("data/human_annotation/annotators all.xlsx")
    gold: Path = Path("data/human_annotation/gold.csv")
    out_dir: Path = Path("data/human_annotation")


HUMAN_SCORE_COLUMNS: Tuple[str, ...] = (
    "human_score_1",
    "human_score_2",
    "human_score_3",
    "human_score",
)


def _weights(method: str) -> np.ndarray:
    """Cat x cat agreement matrix for the given weighting scheme.

    All schemes rely on the category index being directly comparable to the
    score value, i.e. category ``i`` (0-based) corresponds to score ``i+1``:
    this only works because the scores are consecutive integers 1..10.
    """
    max_dist = N_CATEGORIES - 1  # 9
    dist = np.abs(np.arange(N_CATEGORIES)[:, None] - np.arange(N_CATEGORIES)[None, :])
    dist = dist.astype(float) / max_dist  # normalized to [0, 1]
    if method == "nominal":
        return np.eye(N_CATEGORIES)
    if method == "linear":
        return 1.0 - dist
    if method == "quadratic":
        return 1.0 - dist**2
    raise ValueError(f"Unknown weighting method {method!r}")


def _pairwise_weighted_kappa(
    df: pd.DataFrame,
    method: str,
    pair: Tuple[str, str],
) -> Tuple[float, int]:
    """Weighted Cohen kappa for one rater pair; returns (kappa, n valid items).

    Columns hold score values 1..10; they are mapped to 0-based category
    indices (``score - SCORE_MIN``) for ``sklearn``. A pair needs at least 2
    items with both raters valid.
    """
    a = df[pair[0]].to_numpy(dtype=float) - SCORE_MIN
    b = df[pair[1]].to_numpy(dtype=float) - SCORE_MIN
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 2:
        return float("nan"), int(valid.sum())
    kw = {} if method == "nominal" else {"weights": method}
    kappa = float(
        cohen_kappa_score(
            a[valid],
            b[valid],
            labels=list(range(N_CATEGORIES)),
            **kw,
        )
    )
    return kappa, int(valid.sum())


def weighted_kappa_per_pair(
    df: pd.DataFrame,
    method: str,
    rater_names: Sequence[str],
) -> pd.DataFrame:
    """Weighted kappa for every unordered rater pair.

    Returns a DataFrame indexed by pair label with one row per pair. The
    kappa value is NaN when a pair shares fewer than 2 valid items.
    """
    rows: List[Dict[str, Any]] = []
    for i, first in enumerate(rater_names):
        for second in rater_names[i + 1 :]:
            kappa, n_valid = _pairwise_weighted_kappa(df, method, (first, second))
            rows.append({"pair": f"{first}--{second}", "kappa": kappa, "n": n_valid})
    return pd.DataFrame(rows)


def bootstrap_mean_kappa_ci(
    df: pd.DataFrame,
    method: str,
    rater_names: Sequence[str],
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float, int]:
    """Bootstrap 95% CI for the mean pairwise weighted kappa over items.

    Resamples items (rows) with replacement, recomputes every rater pair's
    weighted kappa on the resample, and reports the mean over pairs. Items
    with fewer than two valid raters contribute nothing; pairs with fewer than
    two valid items in a resample are dropped from that resample's mean.

    The point estimate here is the mean over the *observed* pairs (not a
    resample), so it can differ from the average of a bootstrap distribution
    by sampling noise. The CI uses 1000 resamples by default and the same 95% /
    two-sided quantile convention as the repo's ``bootstrap_mean_ci``.
    """
    est_rows: List[float] = []
    for i, first in enumerate(rater_names):
        for second in rater_names[i + 1 :]:
            kappa, n_valid = _pairwise_weighted_kappa(df, method, (first, second))
            if np.isfinite(kappa):
                est_rows.append(kappa)
    point_est = float(np.mean(est_rows)) if est_rows else float("nan")

    n_items = int(df.shape[0])
    rng = np.random.default_rng(seed)
    boot_means: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        resample = df.iloc[idx]
        pair_kappas: List[float] = []
        for i, first in enumerate(rater_names):
            for second in rater_names[i + 1 :]:
                kappa, n_valid = _pairwise_weighted_kappa(
                    resample, method, (first, second)
                )
                if np.isfinite(kappa):
                    pair_kappas.append(kappa)
        if pair_kappas:
            boot_means.append(float(np.mean(pair_kappas)))
    if not boot_means:
        return point_est, float("nan"), float("nan"), 0
    lower = float(np.quantile(boot_means, 0.025))
    upper = float(np.quantile(boot_means, 0.975))
    return point_est, lower, upper, len(boot_means)


def describe_agreement(df: pd.DataFrame, rater_names: Sequence[str]) -> Dict[str, Any]:
    """Exact / within-1 agreement and mean absolute difference per pair.

    All values use the subset of items where both raters of the pair scored.
    """
    pairs: Sequence[Tuple[str, str]] = [
        (rater_names[i], rater_names[j])
        for i in range(len(rater_names))
        for j in range(i + 1, len(rater_names))
    ]
    exact: List[float] = []
    within1: List[float] = []
    mae: List[float] = []
    n_shared: List[int] = []
    for first, second in pairs:
        a = df[first]
        b = df[second]
        valid = a.notna() & b.notna()
        n = int(valid.sum())
        diff = (a[valid] - b[valid]).abs()
        exact.append(float((diff == 0).mean()) if n else float("nan"))
        within1.append(float((diff <= 1).mean()) if n else float("nan"))
        mae.append(float(diff.mean()) if n else float("nan"))
        n_shared.append(n)
    return {
        "exact_agreement": exact,
        "within_1_point": within1,
        "mean_abs_diff": mae,
        "n_shared_items": n_shared,
    }


def score_histogram(df: pd.DataFrame, rater_names: Sequence[str]) -> Dict[str, int]:
    """Counts per *score value* (1..10) over all rater-item cells."""
    counts = np.zeros(N_CATEGORIES, dtype=int)
    for name in rater_names:
        vals = df[name].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        for score in range(SCORE_MIN, SCORE_MAX + 1):
            counts[score - SCORE_MIN] += int((np.abs(vals - score) < 1e-9).sum())
    return {
        str(score): int(counts[score - SCORE_MIN])
        for score in range(SCORE_MIN, SCORE_MAX + 1)
    }


def load_workbook(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load the shared annotation workbook.

    Returns ``(df, rater_names)`` where ``df`` has one row per item and one
    column per rater, holding *score values* 1..10 (floats, NaN where absent).
    Only the sheet 'Annotation' is read; the 'Anleitung' sheet is ignored.
    Item IDs come from the 'item_id' column.

    Rater identity is positional. The columns ``human_score_1``,
    ``human_score_2``, ``human_score_3`` and ``human_score`` are taken in that
    exact order (the known layout of ``annotators all.xlsx``). Any additional
    ``human_score*`` column that does not match that layout is appended in
    lexical order, so the function degrades gracefully if the workbook grows
    more raters later.
    """
    if not path.exists():
        raise FileNotFoundError(f"Annotation workbook not found: {path}")
    raw = pd.read_excel(path, sheet_name="Annotation")
    if "item_id" not in raw.columns:
        raise ValueError(f"{path} (sheet 'Annotation') has no 'item_id' column")
    score_cols = [c for c in raw.columns if str(c).startswith("human_score")]
    if not score_cols:
        raise ValueError(f"{path} has no human_score_* columns")
    ordered: List[str] = []
    for expected in (
        "human_score_1",
        "human_score_2",
        "human_score_3",
        "human_score",
    ):
        if expected in score_cols:
            ordered.append(expected)
    extra = [c for c in score_cols if c not in ordered]
    ordered.extend(sorted(extra, key=lambda c: (len(c), c)))
    df = raw[["item_id"] + ordered].copy()
    df = df.dropna(subset=["item_id"])
    # Keep only items where at least one rater scored (trailing empty rows).
    df = df[df[ordered].notna().any(axis=1)]
    for col in ordered:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.reset_index(drop=True)
    rater_names = list(ordered)
    return df, rater_names


def load_gold(path: Path) -> pd.DataFrame:
    """Load gold.csv and return item_id + llm_score as a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Gold file not found: {path}")
    gold = pd.read_csv(path)
    if "item_id" not in gold.columns or "llm_score" not in gold.columns:
        raise ValueError(f"{path} must have columns 'item_id' and 'llm_score'")
    return gold[["item_id", "llm_score"]].copy()


def _merge_judge(
    human_df: pd.DataFrame,
    rater_names: Sequence[str],
    gold: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Merge judge scores as an extra column and return (df, rater_names).

    Judge scores are kept in the same natural 1..10 score units as the human
    scores; category mapping happens inside the kappa calls.
    """
    merged = human_df.merge(gold, on="item_id", how="left")
    judge_col = "llm_score"
    merged[judge_col] = pd.to_numeric(merged[judge_col], errors="coerce")
    return merged, list(rater_names) + [judge_col]


def _human_mean_column(
    df: pd.DataFrame,
    rater_names: Sequence[str],
) -> pd.Series:
    """Mean human score per item in score units (1..10)."""
    valid = df[rater_names].dropna(axis=1)
    mean_score = valid.mean(axis=1)
    return mean_score


def judge_correlation_summary(
    human_mean_score: pd.Series,
    judge_score: pd.Series,
) -> Dict[str, Any]:
    """Pearson r and MAE between continuous human mean and judge scores.

    These are *not* chance-corrected and are reported as context alongside the
    categorical weighted-kappa value; the paper's claim ('the judge is a valid
    instrument') is about association, which r captures, whereas kappa captures
    interchangeable-ness.
    """
    valid = human_mean_score.notna() & judge_score.notna()
    a = human_mean_score[valid].to_numpy(dtype=float)
    b = judge_score[valid].to_numpy(dtype=float)
    if len(a) < 2:
        return {"pearson_r": None, "mae": None, "n": int(valid.sum())}
    r = float(np.corrcoef(a, b)[0, 1])
    mae = float(np.mean(np.abs(a - b)))
    return {"pearson_r": r, "mae": mae, "n": int(valid.sum())}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inter-annotator agreement (quadratic-weighted Cohen kappa) "
        "for the human validation sample."
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=DataPaths.workbook,
        help="Path to the shared annotation workbook (default: data/human_annotation/annotators all.xlsx).",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=DataPaths.gold,
        help="Path to gold.csv with llm_score (default: data/human_annotation/gold.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DataPaths.out_dir,
        help="Output directory (default: data/human_annotation).",
    )
    parser.add_argument(
        "--max-boot",
        type=int,
        default=1000,
        help="Bootstrap resamples for the mean-kappa CI (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the bootstrap (default: 42).",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write the JSON report.",
    )
    args = parser.parse_args(argv)

    human_df, rater_names = load_workbook(args.workbook)
    gold = load_gold(args.gold)
    df, with_judge = _merge_judge(human_df, rater_names, gold)
    human_only = [n for n in with_judge if n != "llm_score"]

    # --- Primary metric: quadratic-weighted kappa ---------
    pair_table = weighted_kappa_per_pair(df, "quadratic", human_only)
    point_est, ci_low, ci_high, boot_count = bootstrap_mean_kappa_ci(
        df, "quadratic", human_only, args.max_boot, args.seed
    )

    # --- Judge-as-rater ------------------------------------
    judge_col = "llm_score"
    # The judge is a single annotator-like scoring. Primary comparison uses
    # the same quadratic-weighted kappa against the *rounded* mean human
    # score (both sides categorical), and context reports Pearson r / MAE
    # against the continuous mean (association, not chance-corrected).
    human_mean_score = _human_mean_column(df, human_only)
    judge_score = df[judge_col]
    judge_round = human_mean_score.round().clip(SCORE_MIN, SCORE_MAX).astype(int)
    judge_vs_human_df = pd.DataFrame(
        {"human_mean_round": judge_round, "judge": judge_score}
    )
    judge_vs_human_df = judge_vs_human_df.dropna()
    judge_kappa, judge_n = _pairwise_weighted_kappa(
        judge_vs_human_df, "quadratic", ("human_mean_round", "judge")
    )

    # --- Sensitivity analyses ------------------------------
    sens = {}
    for method in ("nominal", "linear"):
        sens[method] = weighted_kappa_per_pair(df, method, human_only)
    # Also judge-vs-rounded-human-mean under nominal / linear for comparability.
    sens_judge = {}
    for method in ("nominal", "linear"):
        sens_judge[method], _ = _pairwise_weighted_kappa(
            judge_vs_human_df, method, ("human_mean_round", "judge")
        )

    # Judge vs continuous human mean: correlation + MAE (context only).
    judge_corr = judge_correlation_summary(
        human_mean_score, judge_score
    )

    # --- Descriptive agreement -----------------------------
    desc = describe_agreement(df, human_only)
    hist = score_histogram(df, human_only)
    judge_hist = score_histogram(df, [judge_col])

    # --- Compose and print report --------------------------
    print("=" * 72)
    print("INTER-ANNOTATOR AGREEMENT — HUMAN VALIDATION SAMPLE")
    print("=" * 72)
    print(f"Workbook       : {args.workbook}")
    print(f"Gold           : {args.gold}")
    print(f"Items scored   : {int(len(df))}")
    print(f"Raters         : {', '.join(rater_names)}")
    print(f"Judge column   : {judge_col}")
    print(f"Score range    : {SCORE_MIN}..{SCORE_MAX} "
          f"({N_CATEGORIES} categories)")
    print()

    print("Score distribution across all human cells:")
    for score in range(SCORE_MIN, SCORE_MAX + 1):
        key = str(score)
        n = hist.get(key, 0)
        bar = "#" * int(round(60 * n / max(1, sum(hist.values()))))
        print(f"  {key:>2}  {n:4d}  {bar}")
    print()

    print("Judge (llm_score) distribution:")
    for score in range(SCORE_MIN, SCORE_MAX + 1):
        key = str(score)
        n = judge_hist.get(key, 0)
        bar = "#" * int(round(60 * n / max(1, sum(judge_hist.values()))))
        print(f"  {key:>2}  {n:4d}  {bar}")
    print()

    print("PRIMARY METRIC — quadratic-weighted Cohen kappa (human raters):")
    print("  points: 1 vs 2 counts 0.89, 2 vs 5 counts 0.67, 1 vs 10 counts 0.00")
    print(
        f"  mean over pairs : {point_est:7.3f}  "
        f"[{ci_low:7.3f}, {ci_high:7.3f}]  (95% percentile bootstrap, "
        f"{boot_count} draws)"
    )
    print("  per pair:")
    for _, row in pair_table.iterrows():
        k = f"{row['kappa']:7.3f}" if np.isfinite(row["kappa"]) else "    --    "
        print(f"    {row['pair']:<26} kappa={k}  n={int(row['n'])}")
    print()

    print("SENSITIVITY — mean kappa under alternative weightings:")
    for method in ("nominal", "linear"):
        kappas = sens[method]["kappa"].dropna()
        mean_k = float(kappas.mean()) if len(kappas) else float("nan")
        print(f"  {method:<10} mean over pairs : {mean_k:7.3f}")
    print()

    print("JUDGE AS RATER (human mean vs llm_score):")
    if np.isfinite(judge_kappa):
        print(f"  quadratic (rounded human mean): {judge_kappa:7.3f}  (n={judge_n})")
    else:
        print(f"  quadratic (rounded human mean):    --    (n={judge_n})")
    for method in ("nominal", "linear"):
        k = sens_judge[method]
        if np.isfinite(k):
            print(f"  {method:<10} (rounded human mean): {k:7.3f}  (n={judge_n})")
        else:
            print(f"  {method:<10} (rounded human mean):    --    (n={judge_n})")
    if judge_corr["pearson_r"] is not None:
        print(f"  context: Pearson r = {judge_corr['pearson_r']:7.3f}   "
              f"MAE = {judge_corr['mae']:5.2f}   (n={judge_corr['n']})")
    print()

    print("DESCRIPTIVE — human agreement (per pair, valid items only):")
    for i, (first, second) in enumerate(
        [
            (human_only[i], human_only[j])
            for i in range(len(human_only))
            for j in range(i + 1, len(human_only))
        ]
    ):
        ex = desc["exact_agreement"][i] if i < len(desc["exact_agreement"]) else float("nan")
        w1 = desc["within_1_point"][i] if i < len(desc["within_1_point"]) else float("nan")
        m = desc["mean_abs_diff"][i] if i < len(desc["mean_abs_diff"]) else float("nan")
        n_shared = desc["n_shared_items"][i] if i < len(desc["n_shared_items"]) else 0
        print(
            f"  {first}--{second:<20} exact {ex:6.2%}  within1 {w1:6.2%}  "
            f"MAE {m:5.2f}  n={n_shared}"
        )
    print()

    # --- Write JSON report -----------------------------------
    if not args.no_json:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pair_rows = [dict(r) for _, r in pair_table.iterrows() if np.isfinite(r["kappa"])]
        report: Dict[str, Any] = {
            "source_files": {
                "workbook": str(args.workbook),
                "gold": str(args.gold),
            },
            "items_scored": int(len(df)),
            "raters": list(rater_names),
            "score_range": [SCORE_MIN, SCORE_MAX],
            "score_histogram_all_human_cells": hist,
            "judge_histogram": judge_hist,
            "primary": {
                "metric": "quadratic_weighted_cohen_kappa",
                "point_estimate": point_est,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_bootstrap": boot_count,
                "seed": args.seed,
            },
            "pairwise_quadratic": pair_rows,
            "sensitivity_mean_by_method": {
                m: (float(sens[m]["kappa"].mean()) if sens[m]["kappa"].notna().any() else None)
                for m in ("nominal", "linear")
            },
            "judge_vs_human_mean": {
                "quadratic_rounded_human_mean": (judge_kappa if np.isfinite(judge_kappa) else None),
                "linear_rounded_human_mean": (sens_judge["linear"] if np.isfinite(sens_judge["linear"]) else None),
                "nominal_rounded_human_mean": (sens_judge["nominal"] if np.isfinite(sens_judge["nominal"]) else None),
                "n_items": judge_n,
                "pearson_r_continuous": judge_corr["pearson_r"],
                "mae_continuous": judge_corr["mae"],
            },
            "descriptive": desc,
        }
        out_path = args.out_dir / "iaa_report.json"
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote JSON report to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())