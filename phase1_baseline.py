"""Phase 1 — Response generation and five-prompt similarity scoring.

For every (party, statement) pair:
  1. Generate a party-condition response and an anti-condition response.
  2. Judge the party response with five isolated single-party prompts.
  3. Use the target party's score for vector weighting.
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
    PARTIES,
    SCHEMA_VERSION,
    JudgeConfig,
    ModelConfig,
    apply_split_manifest,
    atomic_write_csv,
    build_anti_prompt,
    build_party_prompt,
    build_thesis_party_reasoning_lookup,
    create_judge_client,
    env_int,
    env_str,
    filter_party_df,
    file_sha256,
    flatten_party_judge_results,
    generate_response,
    judge_all_parties_similarity_batch,
    load_model_and_tokenizer,
    load_wahlomat_data,
    party_keys_from_env,
    require_schema_columns,
    sample_complete_theses,
    validate_or_create_checkpoint_metadata,
)


def _judge_and_finalize(
    pending,
    judge_client,
    judge_cfg,
    batch_size,
    party_keys,
    reasoning_lookup,
):
    items = [
        (r["thesis"], r["response_party"], r["thesis_key"])
        for r in pending
    ]
    judge_results = judge_all_parties_similarity_batch(
        judge_client,
        judge_cfg,
        items,
        party_keys,
        reasoning_lookup,
        batch_size=batch_size,
    )
    records = []
    for record, jres in zip(pending, judge_results):
        score_cols = flatten_party_judge_results(jres, party_keys)
        target_score = jres.get("scores", {}).get(record["party_key"])
        records.append(
            {
                **record,
                **score_cols,
                "similarity_score": target_score,
                "judge_rationale": jres.get("rationales", {}).get(record["party_key"], ""),
            }
        )
    return records


def run_baseline_study(
    df,
    party_keys,
    model,
    tokenizer,
    cfg,
    judge_client,
    judge_cfg,
    batch_size,
    checkpoint_interval,
    checkpoint_dir: Path,
    resume_metadata: dict,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"phase1_checkpoint_{cfg.short_name}.csv"
    metadata_path = checkpoint_path.with_suffix(".metadata.json")
    validate_or_create_checkpoint_metadata(
        metadata_path,
        resume_metadata,
        "Phase 1",
        checkpoint_exists=checkpoint_path.exists(),
    )
    reasoning_lookup = build_thesis_party_reasoning_lookup(df, party_keys)

    all_records: List[dict] = []
    done_keys = set()
    if checkpoint_path.exists():
        prior = pd.read_csv(checkpoint_path)
        require_schema_columns(
            prior,
            ["schema_version", "party_key", "party_statement_key", "thesis_key", "split"],
            "Phase 1 checkpoint",
        )
        if set(prior["schema_version"].dropna().astype(int)) != {SCHEMA_VERSION}:
            raise ValueError(
                f"Phase 1 checkpoint must contain only schema_version={SCHEMA_VERSION}."
            )
        duplicate_keys = prior.loc[
            prior["party_statement_key"].duplicated(keep=False), "party_statement_key"
        ].unique()
        for key in duplicate_keys:
            if len(prior[prior["party_statement_key"] == key].drop_duplicates()) != 1:
                raise ValueError(
                    f"Phase 1 checkpoint has conflicting records for {key}. "
                    "Clear the checkpoint before rerunning."
                )
        normalized = prior.drop_duplicates("party_statement_key", keep="last")
        expected_pairs = set(
            zip(
                df["party_key"].astype(str),
                df["party_statement_key"].astype(str),
            )
        )
        actual_pairs = set(
            zip(
                normalized["party_key"].astype(str),
                normalized["party_statement_key"].astype(str),
            )
        )
        if not actual_pairs.issubset(expected_pairs):
            raise ValueError(
                "Phase 1 checkpoint contains records outside the current sampled "
                "configuration. Clear the checkpoint before rerunning."
            )
        if set(normalized["split"]) != {"train"}:
            raise ValueError("Phase 1 checkpoint must contain train rows only.")
        if len(normalized) != len(prior):
            atomic_write_csv(checkpoint_path, normalized)
        prior = normalized
        all_records = prior.to_dict("records")
        done_keys = {(r["party_key"], r["party_statement_key"]) for r in all_records}
        print(f"Resuming: {len(all_records)} records from checkpoint.", flush=True)
    else:
        print("No checkpoint — starting fresh.", flush=True)

    pending: List[dict] = []
    stmt_count = len(done_keys)

    def _flush(label=""):
        nonlocal pending
        if not pending:
            return
        print(f"\nJudging {len(pending)} responses{label} ...", flush=True)
        finished = _judge_and_finalize(
            pending,
            judge_client,
            judge_cfg,
            batch_size,
            party_keys,
            reasoning_lookup,
        )
        all_records.extend(finished)
        pending = []
        atomic_write_csv(checkpoint_path, pd.DataFrame(all_records))
        print(f"  Checkpoint -> {checkpoint_path} ({len(all_records)} total)", flush=True)

    for party_key in party_keys:
        party_name = PARTIES[party_key]
        party_df = filter_party_df(df, party_name)
        if party_df.empty:
            print(f"  No data for {party_name}, skipping.", flush=True)
            continue

        for idx, row in tqdm(
            party_df.iterrows(),
            total=len(party_df),
            desc=f"Phase 1 — {party_name}",
        ):
            party_statement_key = row["party_statement_key"]
            if (party_key, party_statement_key) in done_keys:
                continue

            thesis = str(row.get("thesis", "") or "")
            thesis_nr = row.get("thesis_nr")
            thesis_key = row.get("thesis_key")
            election_id = row.get("election_id")
            gt = row.get("true_stance")
            party_reasoning = str(row.get("party_reasoning", "") or "")

            party_prompt = build_party_prompt(party_name, thesis)
            anti_prompt = build_anti_prompt(party_name, thesis)
            response_party = generate_response(model, tokenizer, party_prompt, cfg)
            response_anti = generate_response(model, tokenizer, anti_prompt, cfg)

            pending.append(
                {
                    "party": party_name,
                    "party_key": party_key,
                    "statement_idx": idx,
                    "election_id": election_id,
                    "thesis_nr": thesis_nr,
                    "thesis_key": thesis_key,
                    "party_statement_key": party_statement_key,
                    "schema_version": SCHEMA_VERSION,
                    "split": row["split"],
                    "major_topic": row["major_topic"],
                    "topic_group": row["topic_group"],
                    "thesis": thesis,
                    "ground_truth_stance": gt,
                    "party_reasoning": party_reasoning,
                    "response_party": response_party,
                    "response_anti": response_anti,
                }
            )

            stmt_count += 1
            if stmt_count % checkpoint_interval == 0:
                _flush(f" (checkpoint at {stmt_count} statements)")

    _flush(" (final)")
    return pd.DataFrame(all_records)


def main() -> int:
    data_path = env_str("DATA_PATH", "wahl-o-mat-clean.xlsx", required=True)
    split_manifest = env_str("SPLIT_MANIFEST", "data/thesis_split_manifest.csv", required=True)
    baseline_dir = env_str("BASELINE_RESULTS_DIR", "results_new")
    checkpoint_dir = env_str("PHASE1_CHECKPOINT_DIR", "checkpoints_new/phase1")
    sample_size = env_int("SAMPLE_SIZE", None)
    batch_size = env_int("JUDGE_BATCH_SIZE", JUDGE_BATCH_SIZE)
    checkpoint_interval = env_int("CHECKPOINT_INTERVAL", 50)

    cfg = ModelConfig.from_env()
    judge_cfg = JudgeConfig.from_env()
    party_keys = party_keys_from_env()

    print(f"Model:    {cfg.model_name} ({cfg.short_name})", flush=True)
    print(f"Device:   {cfg.device}", flush=True)
    print(f"Judge:    {judge_cfg.model}", flush=True)
    print(f"Parties:  {party_keys}", flush=True)
    print(f"Tokens:   {cfg.max_new_tokens}", flush=True)

    df = load_wahlomat_data(data_path, parties=party_keys)
    df = apply_split_manifest(df, split_manifest, split="train")
    before_sample = len(df)
    df = sample_complete_theses(df, sample_size)
    if len(df) < before_sample:
        print(
            f"Sampled to {len(df)} complete party-statement rows across "
            f"{df['thesis_key'].nunique()} train theses.",
            flush=True,
        )

    resume_metadata = {
        "phase": 1,
        "schema_version": SCHEMA_VERSION,
        "model": cfg.model_name,
        "model_short_name": cfg.short_name,
        "party_keys": party_keys,
        "data_path": str(Path(data_path)),
        "data_sha256": file_sha256(data_path),
        "manifest_path": str(Path(split_manifest)),
        "manifest_sha256": file_sha256(split_manifest),
        "sample_size": sample_size,
        "sampled_thesis_keys": sorted(df["thesis_key"].astype(str).unique().tolist()),
        "generation": {
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "do_sample": cfg.do_sample,
            "load_in_4bit": cfg.load_in_4bit,
            "load_in_8bit": cfg.load_in_8bit,
        },
        "judge": {
            "model": judge_cfg.model,
            "base_url": judge_cfg.base_url,
            "max_new_tokens": judge_cfg.max_new_tokens,
            "temperature": judge_cfg.temperature,
            "batch_size": batch_size,
            "party_keys": party_keys,
        },
    }

    checkpoint_path = (
        Path(checkpoint_dir) / f"phase1_checkpoint_{cfg.short_name}.csv"
    )
    validate_or_create_checkpoint_metadata(
        checkpoint_path.with_suffix(".metadata.json"),
        resume_metadata,
        "Phase 1",
        checkpoint_exists=checkpoint_path.exists(),
    )

    model, tokenizer = load_model_and_tokenizer(cfg)
    judge_client = create_judge_client(judge_cfg)

    baseline_df = run_baseline_study(
        df,
        party_keys,
        model,
        tokenizer,
        cfg,
        judge_client,
        judge_cfg,
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=Path(checkpoint_dir),
        resume_metadata=resume_metadata,
    )

    out_dir = Path(baseline_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baseline_results_{cfg.short_name}.csv"
    atomic_write_csv(out_path, baseline_df)
    print(f"\nResults -> {out_path}", flush=True)

    print("=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)
    valid = baseline_df["similarity_score"].dropna()
    if len(valid):
        print(f"  Mean target-party similarity: {valid.mean():.2f} / 10")
        print(f"  Median target-party similarity: {valid.median():.2f} / 10")
        print(f"  Scored statements: {len(valid)}/{len(baseline_df)}")
    else:
        print("  No valid similarity scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
