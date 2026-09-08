"""Phase 1b — Neutral baseline (no party prompting).

Generates one neutral-analyst response per unique thesis_key and judges it
with five isolated single-party prompts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    JUDGE_BATCH_SIZE,
    SCHEMA_VERSION,
    JudgeConfig,
    ModelConfig,
    build_neutral_statement_prompt,
    build_thesis_party_reasoning_lookup,
    create_judge_client,
    env_int,
    env_str,
    flatten_party_judge_results,
    generate_response,
    generation_max_tokens_from_env,
    judge_all_parties_similarity_batch,
    load_model_and_tokenizer,
    load_wahlomat_data,
    party_keys_from_env,
    require_schema_columns,
    sample_complete_theses,
)


def run_neutral_baseline(
    df: pd.DataFrame,
    party_keys: List[str],
    model,
    tokenizer,
    cfg: ModelConfig,
    judge_client,
    judge_cfg: JudgeConfig,
    batch_size: int,
    max_new_tokens: int,
    checkpoint_dir: Path,
    checkpoint_interval: int = 50,
) -> pd.DataFrame:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"neutral_baseline_checkpoint_{cfg.short_name}.csv"

    reasoning_lookup = build_thesis_party_reasoning_lookup(df, party_keys)
    unique_df = df.drop_duplicates(subset=["thesis_key"]).reset_index(drop=True)
    print(f"Neutral baseline on {len(unique_df)} unique thesis_key values ...", flush=True)

    all_records: List[dict] = []
    done_thesis_keys = set()
    if checkpoint_path.exists():
        prior = pd.read_csv(checkpoint_path)
        require_schema_columns(
            prior,
            ["schema_version", "thesis_key"],
            "Neutral baseline checkpoint",
        )
        all_records = prior.to_dict("records")
        done_thesis_keys = {r["thesis_key"] for r in all_records}
        print(f"Resuming: {len(all_records)} records from checkpoint.", flush=True)

    pending: List[dict] = []
    generated_count = len(done_thesis_keys)

    def _flush(label=""):
        nonlocal pending
        if not pending:
            return

        judge_items = [
            (r["thesis"], r["neutral_response"], r["thesis_key"])
            for r in pending
        ]
        print(f"\nJudging {len(judge_items)} neutral responses{label} ...", flush=True)
        judge_results = judge_all_parties_similarity_batch(
            judge_client,
            judge_cfg,
            judge_items,
            party_keys,
            reasoning_lookup,
            batch_size=batch_size,
        )

        for record, jres in zip(pending, judge_results):
            score_cols = flatten_party_judge_results(jres, party_keys)
            all_records.append(
                {
                    **record,
                    **score_cols,
                }
            )

        pending = []
        pd.DataFrame(all_records).to_csv(checkpoint_path, index=False)
        print(f"  Checkpoint -> {checkpoint_path} ({len(all_records)} total)", flush=True)

    for _, row in tqdm(unique_df.iterrows(), total=len(unique_df), desc="Neutral baseline"):
        thesis_key = row["thesis_key"]
        if thesis_key in done_thesis_keys:
            continue

        thesis = str(row.get("thesis", "") or "")
        prompt = build_neutral_statement_prompt(thesis)
        response = generate_response(model, tokenizer, prompt, cfg, max_new_tokens=max_new_tokens)
        pending.append(
            {
                "election_id": row.get("election_id"),
                "thesis_nr": row.get("thesis_nr"),
                "thesis_key": thesis_key,
                "schema_version": SCHEMA_VERSION,
                "thesis": thesis,
                "neutral_response": response,
            }
        )

        generated_count += 1
        if generated_count % checkpoint_interval == 0:
            _flush(f" (checkpoint at {generated_count} theses)")

    _flush(" (final)")
    return pd.DataFrame(all_records)


def main() -> int:
    data_path = env_str("DATA_PATH", "wahl-o-mat-clean.xlsx", required=True)
    baseline_dir = env_str("BASELINE_RESULTS_DIR", "results_new")
    checkpoint_dir = env_str(
        "PHASE1_NEUTRAL_CHECKPOINT_DIR",
        "checkpoints_new/phase1_neutral",
    )
    sample_size = env_int("SAMPLE_SIZE", None)
    batch_size = env_int("JUDGE_BATCH_SIZE", JUDGE_BATCH_SIZE)
    checkpoint_interval = env_int("CHECKPOINT_INTERVAL", 50)
    neutral_mnt = env_int("NEUTRAL_MAX_NEW_TOKENS", generation_max_tokens_from_env())

    cfg = ModelConfig.from_env()
    judge_cfg = JudgeConfig.from_env()
    party_keys = party_keys_from_env()

    print(f"Model:    {cfg.model_name} ({cfg.short_name})", flush=True)
    print(f"Device:   {cfg.device}", flush=True)
    print(f"Judge:    {judge_cfg.model}", flush=True)
    print(f"Parties:  {party_keys}", flush=True)
    print(f"Tokens:   {neutral_mnt}", flush=True)

    model, tokenizer = load_model_and_tokenizer(cfg)
    judge_client = create_judge_client(judge_cfg)

    df = load_wahlomat_data(data_path, parties=party_keys)
    before_sample = len(df)
    df = sample_complete_theses(df, sample_size)
    if len(df) < before_sample:
        print(
            f"Sampled to {len(df)} complete party-statement rows across "
            f"{df['thesis_key'].nunique()} theses.",
            flush=True,
        )

    neutral_df = run_neutral_baseline(
        df,
        party_keys,
        model,
        tokenizer,
        cfg,
        judge_client,
        judge_cfg,
        batch_size=batch_size,
        max_new_tokens=neutral_mnt,
        checkpoint_dir=Path(checkpoint_dir),
        checkpoint_interval=checkpoint_interval,
    )

    out_dir = Path(baseline_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"neutral_baseline_{cfg.short_name}_crossparty.csv"
    neutral_df.to_csv(out_path, index=False)
    print(f"\nResults -> {out_path}", flush=True)

    print("=" * 70)
    print("NEUTRAL BASELINE SUMMARY")
    print("=" * 70)
    for pk in party_keys:
        col = f"sim_{pk}"
        if col not in neutral_df.columns:
            continue
        valid = neutral_df[col].dropna()
        if len(valid):
            print(f"  Mean sim vs {pk}: {valid.mean():.2f} / 10 ({len(valid)} theses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
