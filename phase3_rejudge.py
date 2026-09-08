"""Re-judge rows whose judge scores are missing (NaN sim_*) after API rate-limit
failures, without any GPU work: stored response texts are re-sent to the judge.

Processes, per model/party, in order:
  1. alpha0 checkpoints (checkpoints_new/steering/alpha0_ckpt_*.csv)
  2. sweep checkpoints (checkpoints_new/steering/sweep_ckpt_*.csv)
     - alpha=0 rows are filled from the refreshed alpha0 cache (same text, no
       extra API calls), matching how run_sweep copies alpha0 scores.
  3. final results in results_new/steering/ (llama3-70b), same two-step logic.
  4. after scores are complete, rebuild sweep_summary and selected_config
     from the current sweep_results + alpha0_baseline files so derived
     stats cannot stay stale after a hole-filling rejudge.

Guarantees:
  - Up to REJUDGE_MAX_PASSES passes per file until no NaN sim cells remain.
  - Paced at one judge batch request per REJUDGE_MIN_INTERVAL seconds
    (default 1.1s -> ~54 req/min) to stay under the 60 req/min per-key limit.
  - After all files: a verification pass re-reads every CSV and exits with
    status 2 if ANY hole remains, so chained (afterok) resume jobs are blocked
    unless the data is provably complete.
  - Idempotent: only NaN sim cells are judged, so reruns skip completed files.
  - Derived summaries/configs are always recomputed from the filled scores.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils import (
    JUDGE_BATCH_SIZE,
    PARTIES,
    JudgeConfig,
    atomic_write_csv,
    build_thesis_party_reasoning_lookup,
    create_judge_client,
    env_float,
    env_int,
    env_str,
    judge_similarity_batch,
    load_wahlomat_data,
    party_keys_from_env,
)

ROOT = Path(os.environ.get("PIPELINE_ROOT", ".")).resolve()
CKPT_DIR = Path(env_str("PHASE3_CHECKPOINT_DIR", "checkpoints_new/steering"))
if not CKPT_DIR.is_absolute():
    CKPT_DIR = ROOT / CKPT_DIR
RESULTS_DIR = ROOT / env_str("RESULTS_DIR", "results_new/steering")

MODELS = [
    "llama-3-2-1b-instruct", "llama-3-2-3b-instruct", "meta-llama-3-8b-instruct",
    "meta-llama-3-70b-instruct", "qwen3-8b", "ministral-8b-instruct-2410",
    "deepseek-llm-7b-chat", "gemma-4-e4b-it",
]

MIN_INTERVAL = env_float("REJUDGE_MIN_INTERVAL", 1.1)
MAX_PASSES = env_int("REJUDGE_MAX_PASSES", 3)


def target_files(party_keys: List[str]) -> List[Path]:
    """Every CSV the re-judge is responsible for (used by processing AND verify)."""
    files: List[Path] = []
    for model in MODELS:
        for party in party_keys:
            files.append(CKPT_DIR / f"alpha0_ckpt_{model}_{party}.csv")
            files.append(CKPT_DIR / f"sweep_ckpt_{model}_{party}.csv")
    for party in party_keys:  # llama3-70b exported finals
        files.append(RESULTS_DIR / f"alpha0_baseline_meta-llama-3-70b-instruct_{party}.csv")
        files.append(RESULTS_DIR / f"sweep_results_meta-llama-3-70b-instruct_{party}.csv")
    return [f for f in files if f.exists()]


def count_holes(df: pd.DataFrame, party_keys: List[str]) -> int:
    sim_cols = [f"sim_{p}" for p in party_keys if f"sim_{p}" in df.columns]
    if not sim_cols:
        return 0
    return int(df[sim_cols].isna().sum().sum())


def paced_judge(
    client,
    judge_cfg: JudgeConfig,
    items: List[Tuple[str, str, str, str]],
    batch_size: int,
    min_interval: float,
) -> List[Dict[str, Any]]:
    """judge_similarity_batch with one API call per chunk and pacing between calls."""
    results: List[Dict[str, Any]] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        t0 = time.monotonic()
        results.extend(judge_similarity_batch(client, judge_cfg, chunk, batch_size=batch_size))
        elapsed = time.monotonic() - t0
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        done = len(results)
        if done % 200 < batch_size:
            print(f"    judged {done}/{len(items)}", flush=True)
    return results


def rejudge_pass(
    df: pd.DataFrame,
    client,
    judge_cfg: JudgeConfig,
    reasoning_lookup: Dict[str, Dict[str, str]],
    party_keys: List[str],
    batch_size: int,
    alpha0_scores: Optional[Dict[str, Dict[str, Tuple[float, str]]]],
) -> int:
    """One pass over a dataframe: alpha0 propagation, then API re-judge of the
    remaining NaN cells. Returns the number of still-missing cells afterwards."""
    # Step 1: alpha=0 rows inherit scores from the refreshed alpha0 cache.
    if alpha0_scores is not None and "alpha" in df.columns:
        base_mask = df["alpha"].astype(float) == 0.0
        filled = 0
        for party in party_keys:
            col, rcol = f"sim_{party}", f"judge_rationale_{party}"
            if col not in df.columns:
                continue
            for i in df.index[base_mask & df[col].isna()]:
                entry = alpha0_scores.get(str(df.at[i, "thesis_key"]), {}).get(party)
                if entry is not None:
                    df.at[i, col] = entry[0]
                    if rcol in df.columns:
                        df.at[i, rcol] = entry[1]
                    filled += 1
        if filled:
            print(f"    filled {filled} alpha=0 cells from alpha0 cache", flush=True)

    # Step 2: re-judge remaining NaN cells via the API, one party at a time.
    for party in party_keys:
        col, rcol = f"sim_{party}", f"judge_rationale_{party}"
        if col not in df.columns:
            continue
        missing_idx = list(df.index[df[col].isna()])
        if not missing_idx:
            continue
        items = []
        for i in missing_idx:
            thesis_key = str(df.at[i, "thesis_key"])
            reasoning = reasoning_lookup.get(thesis_key, {}).get(party)
            if reasoning is None:
                raise ValueError(f"No reasoning for thesis_key={thesis_key} party={party}")
            items.append((str(df.at[i, "thesis"]), PARTIES[party], reasoning,
                          str(df.at[i, "steered_response"])))
        print(f"    re-judging {len(items)} cells for {party}", flush=True)
        results = paced_judge(client, judge_cfg, items, batch_size, MIN_INTERVAL)
        for i, res in zip(missing_idx, results):
            if res.get("score") is not None:
                df.at[i, col] = res["score"]
                if rcol in df.columns:
                    df.at[i, rcol] = res.get("rationale", "")
    return count_holes(df, party_keys)


def rejudge_csv(
    path: Path,
    client,
    judge_cfg: JudgeConfig,
    reasoning_lookup: Dict[str, Dict[str, str]],
    party_keys: List[str],
    batch_size: int,
    alpha0_scores: Optional[Dict[str, Dict[str, Tuple[float, str]]]] = None,
) -> Dict[str, Dict[str, Tuple[float, str]]]:
    """Fill NaN sim cells in one CSV (up to MAX_PASSES passes, saved per pass).
    Returns the file's score map {thesis_key: {party: (score, rationale)}} for
    downstream alpha=0 fills."""
    df = pd.read_csv(path)
    holes = count_holes(df, party_keys)
    if holes == 0:
        print(f"  {path.name}: complete, skipping", flush=True)
    for pass_no in range(1, MAX_PASSES + 1):
        if holes == 0:
            break
        print(f"  {path.name}: pass {pass_no}/{MAX_PASSES}, {holes} holes", flush=True)
        holes = rejudge_pass(
            df, client, judge_cfg, reasoning_lookup, party_keys, batch_size, alpha0_scores
        )
        atomic_write_csv(path, df)
        print(f"  {path.name}: saved, {holes} holes remain after pass {pass_no}", flush=True)
    if holes:
        print(f"  WARNING {path.name}: {holes} holes after {MAX_PASSES} passes", flush=True)

    score_map: Dict[str, Dict[str, Tuple[float, str]]] = {}
    for _, row in df.iterrows():
        tk = str(row["thesis_key"])
        if tk in score_map:
            continue
        per_party = {}
        for party in party_keys:
            col, rcol = f"sim_{party}", f"judge_rationale_{party}"
            if col in df.columns and pd.notna(row[col]):
                per_party[party] = (float(row[col]), str(row.get(rcol, "") or ""))
        if per_party:
            score_map[tk] = per_party
    return score_map


def refresh_derived_artifacts(party_keys: List[str]) -> int:
    """Rebuild sweep_summary and selected_config from current complete scores."""
    from phase3_steering import refresh_derived_steering_artifacts

    print("\n=== refresh derived summaries/configs ===", flush=True)
    refreshed = 0
    for model in MODELS:
        for party in party_keys:
            results_path = RESULTS_DIR / f"sweep_results_{model}_{party}.csv"
            alpha0_path = RESULTS_DIR / f"alpha0_baseline_{model}_{party}.csv"
            if not (results_path.exists() and alpha0_path.exists()):
                continue
            best = refresh_derived_steering_artifacts(
                RESULTS_DIR, model, party, party_keys
            )
            refreshed += 1
            print(
                f"  {model} {party}: L{best['layer']} a{best['alpha']} "
                f"delta={best['selection_mean_delta']:+.3f} "
                f"n_pairs={best['selection_n_pairs']}",
                flush=True,
            )
    print(f"refreshed {refreshed} model/party derived artifacts", flush=True)
    return refreshed


def verify_all(party_keys: List[str]) -> int:
    """Re-read every target CSV from disk and report remaining holes per file,
    broken down per layer where applicable. Returns total missing cells."""
    print("\n=== verification ===", flush=True)
    total = 0
    for path in target_files(party_keys):
        df = pd.read_csv(path)
        holes = count_holes(df, party_keys)
        total += holes
        status = "OK " if holes == 0 else "HOLES"
        detail = ""
        if holes and "layer" in df.columns:
            sim_cols = [f"sim_{p}" for p in party_keys if f"sim_{p}" in df.columns]
            per_layer = df.assign(_h=df[sim_cols].isna().sum(axis=1)).groupby("layer")["_h"].sum()
            detail = " per-layer: " + ", ".join(f"L{int(l)}={int(h)}" for l, h in per_layer.items() if h)
        print(f"  [{status}] {path.name}: {holes} holes{detail}", flush=True)
    print(f"verification total: {total} missing cells", flush=True)
    return total


def main() -> None:
    if "--verify-only" in sys.argv:
        sys.exit(0 if verify_all(party_keys_from_env()) == 0 else 2)
    if "--refresh-derived" in sys.argv:
        refresh_derived_artifacts(party_keys_from_env())
        sys.exit(0)

    party_keys = party_keys_from_env()
    batch_size = env_int("JUDGE_BATCH_SIZE", JUDGE_BATCH_SIZE)
    judge_cfg = JudgeConfig.from_env()
    client = create_judge_client(judge_cfg)
    data_path = ROOT / env_str("DATA_PATH", "data/wahl-o-mat-clean.xlsx")
    df = load_wahlomat_data(str(data_path), parties=party_keys)
    reasoning_lookup = build_thesis_party_reasoning_lookup(df, party_keys)
    print(f"Re-judge: pacing 1 call / {MIN_INTERVAL}s, batch {batch_size}, "
          f"max {MAX_PASSES} passes/file", flush=True)

    for model in MODELS:
        print(f"== {model} ==", flush=True)
        for party in party_keys:
            alpha0_path = CKPT_DIR / f"alpha0_ckpt_{model}_{party}.csv"
            sweep_path = CKPT_DIR / f"sweep_ckpt_{model}_{party}.csv"
            alpha0_scores = None
            if alpha0_path.exists():
                alpha0_scores = rejudge_csv(
                    alpha0_path, client, judge_cfg, reasoning_lookup, party_keys, batch_size
                )
            if sweep_path.exists():
                rejudge_csv(
                    sweep_path, client, judge_cfg, reasoning_lookup, party_keys, batch_size,
                    alpha0_scores=alpha0_scores,
                )

    model = "meta-llama-3-70b-instruct"
    print(f"== {model} final results ==", flush=True)
    for party in party_keys:
        alpha0_path = RESULTS_DIR / f"alpha0_baseline_{model}_{party}.csv"
        sweep_path = RESULTS_DIR / f"sweep_results_{model}_{party}.csv"
        alpha0_scores = None
        if alpha0_path.exists():
            alpha0_scores = rejudge_csv(
                alpha0_path, client, judge_cfg, reasoning_lookup, party_keys, batch_size
            )
        if sweep_path.exists():
            rejudge_csv(
                sweep_path, client, judge_cfg, reasoning_lookup, party_keys, batch_size,
                alpha0_scores=alpha0_scores,
            )

    remaining = verify_all(party_keys)
    if remaining:
        print(f"FAILED: {remaining} judge cells still missing", flush=True)
        sys.exit(2)
    refresh_derived_artifacts(party_keys)
    print("Re-judge complete: all target files have full judge coverage.", flush=True)


if __name__ == "__main__":
    main()
