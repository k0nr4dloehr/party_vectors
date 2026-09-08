"""Phase 3 — Causal intervention with five-prompt cross-party scoring."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
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
    atomic_write_json,
    bootstrap_mean_ci,
    build_neutral_statement_prompt,
    build_thesis_party_reasoning_lookup,
    create_judge_client,
    env_int,
    env_list_float,
    env_list_int,
    env_str,
    file_sha256,
    flatten_party_judge_results,
    generate_response,
    generation_max_tokens_from_env,
    get_num_layers,
    get_residual_stream_layers,
    judge_all_parties_similarity_batch,
    load_ideology_vectors,
    load_model_and_tokenizer,
    load_wahlomat_data,
    party_keys_from_env,
    require_schema_columns,
    sample_complete_theses,
    select_best_steering_config,
    validate_or_create_checkpoint_metadata,
)


def make_batched_hook(vector: torch.Tensor, alphas: List[float]) -> Callable:
    """Arditi-style steering: add alpha * unit(vec) at every token position.

    The vector is L2-normalized so a given alpha means the same residual-stream
    perturbation magnitude at every layer. The hook fires once on the full
    prompt during prefill and once per decode step, adding the normalized
    vector to all positions in both cases (prompt and generated tokens).
    """
    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        vec = vector.to(device=hs.device, dtype=hs.dtype)
        vec = vec / vec.norm().clamp(min=1e-12)
        alpha_tensor = torch.tensor(alphas, device=hs.device, dtype=hs.dtype).view(-1, 1, 1)
        hs = hs + alpha_tensor * vec
        return (hs,) + output[1:] if is_tuple else hs

    return hook


def generate_batch_with_steer(
    model,
    tokenizer,
    cfg: ModelConfig,
    layers_module,
    prompt: str,
    layer: int,
    vector: torch.Tensor,
    alphas: List[float],
    max_new_tokens: int,
) -> List[str]:
    if not alphas:
        return []

    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
    batch_size = len(alphas)
    ids = inputs["input_ids"].repeat(batch_size, 1).to(cfg.device)
    mask = inputs["attention_mask"].repeat(batch_size, 1).to(cfg.device)

    hook = make_batched_hook(vector, alphas)
    handle = layers_module[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                do_sample=cfg.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    input_length = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(outputs[i][input_length:], skip_special_tokens=True).strip()
        for i in range(batch_size)
    ]


def _record_from_row(
    row: pd.Series,
    party_key: str,
    party_name: str,
    layer: int,
    alpha: float,
    response: str,
    split: str,
) -> dict:
    return {
        "evaluation_key": f"{row.get('thesis_key')}::L{layer}::A{alpha:g}",
        "steered_party": party_name,
        "steered_party_key": party_key,
        "layer": layer,
        "alpha": alpha,
        "statement_idx": row.get("statement_idx"),
        "election_id": row.get("election_id"),
        "thesis_nr": row.get("thesis_nr"),
        "thesis_key": row.get("thesis_key"),
        "party_statement_key": row.get("party_statement_key"),
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "thesis": row.get("thesis"),
        "steered_response": response,
    }


def _evaluation_key(thesis_key: str, layer: int, alpha: float) -> str:
    return f"{thesis_key}::L{layer}::A{alpha:g}"


def expected_layer_evaluation_keys(
    sweep_df: pd.DataFrame,
    layer: int,
    alphas: List[float],
) -> set[str]:
    return {
        _evaluation_key(str(thesis_key), layer, alpha)
        for thesis_key in sweep_df["thesis_key"]
        for alpha in alphas
    }


def require_requested_vector_layers(
    vectors: Dict[int, torch.Tensor],
    layers: List[int],
    model_short_name: str,
    party_key: str,
) -> None:
    missing_layers = sorted(set(layers) - set(vectors))
    if missing_layers:
        raise FileNotFoundError(
            "Requested steering vector layers are missing for "
            f"{model_short_name}/{party_key}: {missing_layers}. "
            "Recompute Phase 2 vectors or correct SWEEP_LAYERS."
        )


def validate_and_normalize_sweep_checkpoint(
    prior: pd.DataFrame,
    sweep_df: pd.DataFrame,
    layers: List[int],
    alphas: List[float],
    context: str = "Phase 3 checkpoint",
) -> tuple[pd.DataFrame, set[int]]:
    """Keep only exact completed layers and reject corrupt or stale records."""
    required = [
        "schema_version",
        "evaluation_key",
        "thesis_key",
        "layer",
        "alpha",
        "split",
    ]
    require_schema_columns(prior, required, context)
    if prior.empty:
        return prior.copy(), set()
    if set(prior["schema_version"].dropna().astype(int)) != {SCHEMA_VERSION}:
        raise ValueError(f"{context}: every row must use schema_version={SCHEMA_VERSION}.")
    if prior[required].isna().any().any():
        raise ValueError(f"{context}: required key fields contain missing values.")

    normalized = prior.copy()
    normalized["thesis_key"] = normalized["thesis_key"].astype(str)
    normalized["layer"] = normalized["layer"].astype(int)
    normalized["alpha"] = normalized["alpha"].astype(float)
    normalized["evaluation_key"] = normalized["evaluation_key"].astype(str)

    duplicate_keys = normalized.loc[
        normalized["evaluation_key"].duplicated(keep=False), "evaluation_key"
    ].unique()
    for key in duplicate_keys:
        group = normalized[normalized["evaluation_key"] == key]
        canonical_rows = {
            json.dumps(
                {
                    column: None if pd.isna(value) else value
                    for column, value in row.items()
                },
                sort_keys=True,
                default=str,
            )
            for row in group.to_dict("records")
        }
        if len(canonical_rows) != 1:
            raise ValueError(
                f"{context}: duplicate evaluation_key '{key}' has conflicting records. "
                "Clear the checkpoint before rerunning."
            )
    normalized = normalized.drop_duplicates("evaluation_key", keep="last")

    requested_layers = set(layers)
    requested_alphas = {float(alpha) for alpha in alphas}
    requested_theses = set(sweep_df["thesis_key"].astype(str))
    if not set(normalized["layer"]).issubset(requested_layers):
        raise ValueError(f"{context}: contains layers outside the requested configuration.")
    if not set(normalized["alpha"]).issubset(requested_alphas):
        raise ValueError(f"{context}: contains alphas outside the requested configuration.")
    if not set(normalized["thesis_key"]).issubset(requested_theses):
        raise ValueError(f"{context}: contains theses outside the selected configuration.")

    expected_key_universe = set().union(
        *(expected_layer_evaluation_keys(sweep_df, layer, alphas) for layer in layers)
    )
    recomputed = normalized.apply(
        lambda row: _evaluation_key(row["thesis_key"], int(row["layer"]), float(row["alpha"])),
        axis=1,
    )
    if not normalized["evaluation_key"].equals(recomputed):
        raise ValueError(f"{context}: evaluation_key does not match thesis/layer/alpha fields.")
    unexpected = set(normalized["evaluation_key"]) - expected_key_universe
    if unexpected:
        raise ValueError(f"{context}: contains unexpected evaluation keys: {sorted(unexpected)[:3]}")

    completed_layers: set[int] = set()
    complete_parts: List[pd.DataFrame] = []
    for layer, group in normalized.groupby("layer", sort=False):
        expected = expected_layer_evaluation_keys(sweep_df, int(layer), alphas)
        actual = set(group["evaluation_key"])
        if actual == expected and len(group) == len(expected):
            completed_layers.add(int(layer))
            complete_parts.append(group)
        else:
            print(
                f"  Discarding incomplete checkpoint layer {int(layer)} "
                f"({len(actual)}/{len(expected)} exact evaluations).",
                flush=True,
            )

    if not complete_parts:
        return normalized.iloc[0:0].copy(), set()
    complete = pd.concat(complete_parts, ignore_index=True)
    return complete, completed_layers


def validate_alpha0_cache(
    alpha0_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    cfg: ModelConfig,
    party_key: str,
    all_party_keys: List[str],
) -> pd.DataFrame:
    required = [
        "schema_version",
        "model",
        "model_short_name",
        "steered_party_key",
        "thesis_key",
        "thesis",
        "split",
        "steered_response",
        *[f"sim_{pk}" for pk in all_party_keys],
    ]
    require_schema_columns(alpha0_df, required, "Phase 3 alpha=0 cache")
    if alpha0_df.empty:
        raise ValueError("Phase 3 alpha=0 cache is empty.")
    if alpha0_df["thesis_key"].duplicated().any():
        raise ValueError("Phase 3 alpha=0 cache contains duplicate thesis_key records.")
    if set(alpha0_df["schema_version"].astype(int)) != {SCHEMA_VERSION}:
        raise ValueError("Phase 3 alpha=0 cache has an invalid schema version.")
    if set(alpha0_df["model"]) != {cfg.model_name}:
        raise ValueError("Phase 3 alpha=0 cache model does not match the current model.")
    if set(alpha0_df["model_short_name"]) != {cfg.short_name}:
        raise ValueError("Phase 3 alpha=0 cache model short name does not match.")
    if set(alpha0_df["steered_party_key"]) != {party_key}:
        raise ValueError("Phase 3 alpha=0 cache party does not match.")
    if set(alpha0_df["split"]) != {"selection"}:
        raise ValueError("Phase 3 alpha=0 cache must contain selection rows only.")

    expected = set(sweep_df["thesis_key"].astype(str))
    actual = set(alpha0_df["thesis_key"].astype(str))
    if actual != expected or len(alpha0_df) != len(expected):
        raise ValueError(
            "Phase 3 alpha=0 cache does not contain the exact selected thesis keys."
        )
    expected_text = sweep_df.set_index("thesis_key")["thesis"].astype(str).to_dict()
    for _, row in alpha0_df.iterrows():
        key = str(row["thesis_key"])
        if str(row["thesis"]) != expected_text[key]:
            raise ValueError(f"Phase 3 alpha=0 cache thesis text changed for {key}.")
        if not str(row["steered_response"]).strip():
            raise ValueError(f"Phase 3 alpha=0 cache has an empty response for {key}.")
    return alpha0_df


def align_judge_results_by_evaluation_key(
    records: List[dict],
    judge_results: List[dict],
) -> Dict[str, dict]:
    """Align positional judge outputs to unique thesis-layer-alpha evaluations."""
    if len(records) != len(judge_results):
        raise RuntimeError(
            f"Judge alignment error: {len(records)} records, "
            f"{len(judge_results)} results."
        )
    keys = [record["evaluation_key"] for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Evaluation keys must be unique before judge alignment.")
    return {
        record["evaluation_key"]: result
        for record, result in zip(records, judge_results)
    }


def build_alpha0_cache(
    model,
    tokenizer,
    cfg: ModelConfig,
    sweep_df: pd.DataFrame,
    reasoning_lookup: Dict[str, Dict[str, str]],
    judge_client,
    judge_cfg: JudgeConfig,
    all_party_keys: List[str],
    judge_batch_size: int,
    max_new_tokens: int,
    party_key: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    pending_records: List[dict] = []
    pending_items: List[tuple] = []

    for _, row in tqdm(sweep_df.iterrows(), total=len(sweep_df), desc="Alpha=0 baseline"):
        thesis = str(row.get("thesis", "") or "")
        prompt = build_neutral_statement_prompt(thesis)
        response = generate_response(model, tokenizer, prompt, cfg, max_new_tokens=max_new_tokens)
        record = {
            "schema_version": SCHEMA_VERSION,
            "model": cfg.model_name,
            "model_short_name": cfg.short_name,
            "steered_party_key": party_key,
            "thesis_key": row["thesis_key"],
            "thesis": thesis,
            "split": row["split"],
            "steered_response": response,
        }
        pending_records.append(record)
        pending_items.append((thesis, response, row["thesis_key"]))

    judge_results = judge_all_parties_similarity_batch(
        judge_client,
        judge_cfg,
        pending_items,
        all_party_keys,
        reasoning_lookup,
        batch_size=judge_batch_size,
    )
    if len(judge_results) != len(pending_records):
        raise RuntimeError(
            "Alpha=0 judge alignment error: "
            f"{len(pending_records)} records, {len(judge_results)} results."
        )
    for record, jres in zip(pending_records, judge_results):
        rows.append({**record, **flatten_party_judge_results(jres, all_party_keys)})
    return pd.DataFrame(rows)


def load_or_build_alpha0_cache(
    cache_path: Path,
    cache_metadata: dict,
    model,
    tokenizer,
    cfg: ModelConfig,
    sweep_df: pd.DataFrame,
    reasoning_lookup: Dict[str, Dict[str, str]],
    judge_client,
    judge_cfg: JudgeConfig,
    all_party_keys: List[str],
    judge_batch_size: int,
    max_new_tokens: int,
    party_key: str,
) -> pd.DataFrame:
    metadata_path = cache_path.with_suffix(".metadata.json")
    validate_or_create_checkpoint_metadata(
        metadata_path,
        cache_metadata,
        "Phase 3 alpha=0 cache",
        checkpoint_exists=cache_path.exists(),
    )
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        try:
            validated = validate_alpha0_cache(
                cached,
                sweep_df,
                cfg,
                party_key,
                all_party_keys,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Phase 3 alpha=0 cache is invalid: {exc} "
                f"Clear {cache_path} and {metadata_path} before rerunning."
            ) from exc
        print(f"Reusing alpha=0 cache -> {cache_path}", flush=True)
        return validated

    built = build_alpha0_cache(
        model,
        tokenizer,
        cfg,
        sweep_df,
        reasoning_lookup,
        judge_client,
        judge_cfg,
        all_party_keys,
        judge_batch_size,
        max_new_tokens,
        party_key,
    )
    validated = validate_alpha0_cache(
        built,
        sweep_df,
        cfg,
        party_key,
        all_party_keys,
    )
    atomic_write_csv(cache_path, validated)
    print(f"Saved alpha=0 cache -> {cache_path}", flush=True)
    return validated


def run_sweep(
    model,
    tokenizer,
    cfg: ModelConfig,
    layers_module,
    sweep_df: pd.DataFrame,
    party_key: str,
    vectors: Dict[int, torch.Tensor],
    layers: List[int],
    alphas: List[float],
    judge_client,
    judge_cfg: JudgeConfig,
    reasoning_lookup: Dict[str, Dict[str, str]],
    all_party_keys: List[str],
    judge_batch_size: int,
    checkpoint_dir: Path,
    sweep_max_new_tokens: int,
    alpha0_df: pd.DataFrame,
    resume_metadata: dict,
) -> pd.DataFrame:
    party_name = PARTIES[party_key]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"sweep_ckpt_{cfg.short_name}_{party_key}.csv"
    metadata_path = checkpoint_path.with_suffix(".metadata.json")

    require_requested_vector_layers(vectors, layers, cfg.short_name, party_key)
    validate_or_create_checkpoint_metadata(
        metadata_path,
        resume_metadata,
        "Phase 3 sweep",
        checkpoint_exists=checkpoint_path.exists(),
    )

    nonzero_alphas = [a for a in alphas if a != 0.0]
    alpha0_map = alpha0_df.set_index("thesis_key")

    all_records: List[dict] = []
    done_layers: set = set()
    if checkpoint_path.exists():
        prior = pd.read_csv(checkpoint_path)
        normalized, done_layers = validate_and_normalize_sweep_checkpoint(
            prior,
            sweep_df,
            layers,
            alphas,
        )
        all_records = normalized.to_dict("records")
        if len(normalized) != len(prior):
            atomic_write_csv(checkpoint_path, normalized)
        print(
            f"  Resuming: {len(done_layers)} layers done, {len(all_records)} records.",
            flush=True,
        )

    remaining = [layer for layer in layers if layer not in done_layers]
    if not remaining:
        print("  All layers done.", flush=True)
        return pd.DataFrame(all_records)

    pbar = tqdm(total=len(remaining) * len(sweep_df), desc=f"Sweep — {party_name}")

    for layer in remaining:
        vec = vectors[layer]
        layer_pending: List[dict] = []

        for _, row in sweep_df.iterrows():
            thesis_key = row["thesis_key"]
            alpha0_row = alpha0_map.loc[thesis_key]
            for alpha in alphas:
                if alpha == 0.0:
                    response = alpha0_row["steered_response"]
                else:
                    continue
                layer_pending.append(
                    _record_from_row(
                        row,
                        party_key,
                        party_name,
                        layer,
                        alpha,
                        response,
                        row["split"],
                    )
                )

            if nonzero_alphas:
                thesis = str(row.get("thesis", "") or "")
                neutral_prompt = build_neutral_statement_prompt(thesis)
                steered_responses = generate_batch_with_steer(
                    model,
                    tokenizer,
                    cfg,
                    layers_module,
                    neutral_prompt,
                    layer,
                    vec,
                    nonzero_alphas,
                    sweep_max_new_tokens,
                )
                if len(steered_responses) != len(nonzero_alphas):
                    raise RuntimeError(
                        f"Layer {layer} generation returned {len(steered_responses)} "
                        f"responses for {len(nonzero_alphas)} requested nonzero alphas "
                        f"at thesis {thesis_key}. Nothing was checkpointed."
                    )
                for alpha, response in zip(nonzero_alphas, steered_responses):
                    layer_pending.append(
                        _record_from_row(
                            row,
                            party_key,
                            party_name,
                            layer,
                            alpha,
                            response,
                            row["split"],
                        )
                    )
            pbar.update(1)

        judge_items = [
            (r["thesis"], r["steered_response"], r["thesis_key"])
            for r in layer_pending
            if r["alpha"] != 0.0
        ]
        nonzero_records = [r for r in layer_pending if r["alpha"] != 0.0]
        print(
            f"\n  Judging layer {layer}: {len(judge_items)} steered responses ...",
            flush=True,
        )
        if judge_items:
            judge_results = judge_all_parties_similarity_batch(
                judge_client,
                judge_cfg,
                judge_items,
                all_party_keys,
                reasoning_lookup,
                batch_size=judge_batch_size,
            )
            judged_by_key = align_judge_results_by_evaluation_key(
                nonzero_records,
                judge_results,
            )
        else:
            judged_by_key = {}

        for record in layer_pending:
            if record["alpha"] == 0.0:
                alpha0_row = alpha0_map.loc[record["thesis_key"]]
                score_cols = {
                    f"sim_{pk}": alpha0_row.get(f"sim_{pk}") for pk in all_party_keys
                }
            else:
                jres = judged_by_key[record["evaluation_key"]]
                score_cols = flatten_party_judge_results(jres, all_party_keys)
            record.update(score_cols)

        layer_df = pd.DataFrame(layer_pending)
        expected_keys = expected_layer_evaluation_keys(sweep_df, layer, alphas)
        actual_keys = set(layer_df["evaluation_key"])
        if (
            len(layer_df) != len(expected_keys)
            or actual_keys != expected_keys
            or layer_df["evaluation_key"].duplicated().any()
        ):
            raise RuntimeError(
                f"Phase 3 produced an invalid layer {layer}: "
                f"{len(layer_df)} rows and {len(actual_keys)} unique keys; "
                f"expected {len(expected_keys)} exact keys. Nothing was checkpointed."
            )
        all_records.extend(layer_df.to_dict("records"))

        checkpoint_df = pd.DataFrame(all_records)
        checkpoint_df, completed = validate_and_normalize_sweep_checkpoint(
            checkpoint_df,
            sweep_df,
            layers,
            alphas,
            context="Phase 3 generated checkpoint",
        )
        if layer not in completed:
            raise RuntimeError(f"Phase 3 layer {layer} failed checkpoint validation.")
        all_records = checkpoint_df.to_dict("records")
        atomic_write_csv(checkpoint_path, checkpoint_df)
        print(f"  Checkpoint -> {checkpoint_path}", flush=True)

    pbar.close()
    final_df, completed = validate_and_normalize_sweep_checkpoint(
        pd.DataFrame(all_records),
        sweep_df,
        layers,
        alphas,
        context="Phase 3 final sweep",
    )
    if completed != set(layers):
        missing = sorted(set(layers) - completed)
        raise RuntimeError(f"Phase 3 final sweep is missing completed layers: {missing}")
    return final_df


def summarise_sweep(
    results_df: pd.DataFrame,
    alpha0_df: pd.DataFrame,
    all_party_keys: List[str],
    steered_party_key: str,
) -> pd.DataFrame:
    rows = []
    alpha0_by_thesis = alpha0_df.set_index("thesis_key")

    for (layer, alpha, split), grp in results_df.groupby(["layer", "alpha", "split"]):
        for pk in all_party_keys:
            col = f"sim_{pk}"
            valid = grp[col].dropna()
            mean_score = float(valid.mean()) if len(valid) else None
            if alpha == 0.0:
                mean_delta = 0.0
                ci_low, ci_high = 0.0, 0.0
                n_pairs = len(valid)
            else:
                deltas = []
                for _, row in grp.iterrows():
                    base = alpha0_by_thesis.loc[row["thesis_key"], col]
                    score = row[col]
                    if pd.isna(base) or pd.isna(score):
                        continue
                    if row["split"] != split:
                        continue
                    deltas.append(float(score) - float(base))
                ci_low, ci_high = bootstrap_mean_ci(deltas)
                mean_delta = float(sum(deltas) / len(deltas)) if deltas else None
                n_pairs = len(deltas)

            rows.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "split": split,
                    "compared_party_key": pk,
                    "n_total": len(grp),
                    "n_valid": len(valid),
                    "n_pairs": n_pairs,
                    "mean_score": round(mean_score, 4) if mean_score is not None else None,
                    "mean_delta": round(mean_delta, 4) if mean_delta is not None else None,
                    "ci_low": round(ci_low, 4) if ci_low is not None else None,
                    "ci_high": round(ci_high, 4) if ci_high is not None else None,
                }
            )

    summary_df = pd.DataFrame(rows)
    return summary_df


def prepare_sweep_dataframe(
    workbook_df: pd.DataFrame,
    split_manifest: str,
    sample_size: Optional[int],
) -> pd.DataFrame:
    selection_df = apply_split_manifest(
        workbook_df,
        split_manifest,
        split="selection",
    )
    sweep_df = selection_df.drop_duplicates("thesis_key").reset_index(drop=True)
    return sample_complete_theses(sweep_df, sample_size)


def resolve_party_key(cli_party: Optional[str], all_party_keys: List[str]) -> str:
    if cli_party:
        if cli_party not in PARTIES:
            raise ValueError(f"Unknown party '{cli_party}'. Valid: {list(PARTIES)}")
        return cli_party
    env_p = os.environ.get("STEER_PARTY")
    if env_p:
        if env_p not in PARTIES:
            raise ValueError(f"Unknown STEER_PARTY '{env_p}'.")
        return env_p
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        i = int(task_id)
        if not (0 <= i < len(all_party_keys)):
            raise IndexError(f"SLURM_ARRAY_TASK_ID={i} out of range for {all_party_keys}")
        return all_party_keys[i]
    raise RuntimeError(
        "No party specified. Set --party, STEER_PARTY env, or SLURM_ARRAY_TASK_ID."
    )


def select_best_config_or_raise(
    results_df: pd.DataFrame,
    alpha0_df: pd.DataFrame,
    party_key: str,
) -> Dict[str, Any]:
    best = select_best_steering_config(
        results_df,
        alpha0_df,
        target_col=f"sim_{party_key}",
        steered_party_key=party_key,
        split="selection",
    )
    if not best:
        raise RuntimeError(
            "Phase 3 could not select a valid steering configuration for "
            f"{party_key}: no valid paired selection-set scores were available. "
            "The array task is failing so dependent Phase 4 jobs will not run."
        )
    return best


def refresh_derived_steering_artifacts(
    results_dir: Path,
    model_short_name: str,
    party_key: str,
    all_party_keys: Optional[List[str]] = None,
    *,
    results_df: Optional[pd.DataFrame] = None,
    alpha0_df: Optional[pd.DataFrame] = None,
    split_manifest: Optional[str] = None,
    phase3_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    """Recompute sweep_summary and selected_config from current score CSVs.

    After a rejudge fills NaN sim cells, the raw sweep/alpha0 files can be
    complete while the derived summary and selected config still describe the
    incomplete pairing. This rebuilds both from the scores on disk (or from
    in-memory frames) and keeps fingerprint/manifest metadata from an existing
    selected_config when the caller does not supply replacements.
    """
    all_party_keys = all_party_keys or list(PARTIES.keys())
    results_dir = Path(results_dir)
    results_path = results_dir / f"sweep_results_{model_short_name}_{party_key}.csv"
    alpha0_path = results_dir / f"alpha0_baseline_{model_short_name}_{party_key}.csv"
    summary_path = results_dir / f"sweep_summary_{model_short_name}_{party_key}.csv"
    config_path = results_dir / f"selected_config_{model_short_name}_{party_key}.json"

    if results_df is None:
        if not results_path.exists():
            raise FileNotFoundError(f"Missing sweep results: {results_path}")
        results_df = pd.read_csv(results_path)
    if alpha0_df is None:
        if not alpha0_path.exists():
            raise FileNotFoundError(f"Missing alpha=0 baseline: {alpha0_path}")
        alpha0_df = pd.read_csv(alpha0_path)

    prior: Dict[str, Any] = {}
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
    if split_manifest is None:
        split_manifest = prior.get("split_manifest")
    if phase3_fingerprint is None:
        phase3_fingerprint = prior.get("phase3_fingerprint")

    summary_df = summarise_sweep(results_df, alpha0_df, all_party_keys, party_key)
    best = select_best_config_or_raise(results_df, alpha0_df, party_key)
    if split_manifest is not None:
        best["split_manifest"] = split_manifest
    if phase3_fingerprint is not None:
        best["phase3_fingerprint"] = phase3_fingerprint

    atomic_write_csv(summary_path, summary_df)
    atomic_write_json(config_path, best)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--party",
        default=None,
        help="Party key to steer. Overrides STEER_PARTY and SLURM_ARRAY_TASK_ID.",
    )
    args = parser.parse_args()

    data_path = env_str("DATA_PATH", "wahl-o-mat-clean.xlsx", required=True)
    split_manifest = env_str("SPLIT_MANIFEST", "data/thesis_split_manifest.csv", required=True)
    vector_dir = env_str("VECTOR_DIR", "vectors_new")
    results_dir = env_str("RESULTS_DIR", "results_new/steering")
    checkpoint_dir = env_str("PHASE3_CHECKPOINT_DIR", "checkpoints_new/steering")
    sample_size = env_int("SAMPLE_SIZE", None)
    sweep_mnt = env_int("SWEEP_MAX_NEW_TOKENS", generation_max_tokens_from_env())
    batch_size = env_int("JUDGE_BATCH_SIZE", JUDGE_BATCH_SIZE)

    cfg = ModelConfig.from_env()
    judge_cfg = JudgeConfig.from_env()
    all_party_keys = party_keys_from_env()
    party_key = resolve_party_key(args.party, all_party_keys)
    party_name = PARTIES[party_key]

    print(f"Model:    {cfg.model_name} ({cfg.short_name})", flush=True)
    print(f"Party:    {party_name} ({party_key})", flush=True)

    model, tokenizer = load_model_and_tokenizer(cfg)
    num_layers = get_num_layers(model)
    layers_module = get_residual_stream_layers(model)

    sweep_layers = env_list_int("SWEEP_LAYERS", list(range(num_layers)))
    sweep_alphas = env_list_float(
        "SWEEP_ALPHAS",
        [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    )
    if not sweep_layers:
        raise ValueError("SWEEP_LAYERS resolved to an empty list.")
    if not sweep_alphas:
        raise ValueError("SWEEP_ALPHAS resolved to an empty list.")
    if len(sweep_layers) != len(set(sweep_layers)):
        raise ValueError("SWEEP_LAYERS must not contain duplicates.")
    if len(sweep_alphas) != len(set(sweep_alphas)):
        raise ValueError("SWEEP_ALPHAS must not contain duplicates.")
    print(
        f"Layers:   {len(sweep_layers)} "
        f"({sweep_layers[:5]}{'...' if len(sweep_layers) > 5 else ''})",
        flush=True,
    )
    print(f"Alphas:   {sweep_alphas}", flush=True)
    print(f"Max new tokens: {sweep_mnt}", flush=True)

    df = load_wahlomat_data(data_path, parties=all_party_keys)
    reasoning_lookup = build_thesis_party_reasoning_lookup(df, all_party_keys)
    sweep_df = prepare_sweep_dataframe(
        df,
        split_manifest,
        sample_size,
    )
    print(
        f"Sweep rows: {len(sweep_df)} "
        f"({sweep_df['thesis_key'].nunique()} unique thesis_key values)",
        flush=True,
    )

    vectors = load_ideology_vectors(vector_dir, cfg.short_name, party_key)
    require_requested_vector_layers(vectors, sweep_layers, cfg.short_name, party_key)

    common_metadata = {
        "schema_version": SCHEMA_VERSION,
        "model": cfg.model_name,
        "model_short_name": cfg.short_name,
        "party_key": party_key,
        "all_party_keys": all_party_keys,
        "data_path": str(Path(data_path)),
        "data_sha256": file_sha256(data_path),
        "manifest_path": str(Path(split_manifest)),
        "manifest_sha256": file_sha256(split_manifest),
        "sample_size": sample_size,
        "selected_thesis_keys": sorted(
            sweep_df["thesis_key"].astype(str).unique().tolist()
        ),
        "selected_theses": {
            str(row["thesis_key"]): str(row["thesis"])
            for _, row in sweep_df.iterrows()
        },
        "generation": {
            "max_new_tokens": sweep_mnt,
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
        },
    }
    vector_party_dir = Path(vector_dir) / cfg.short_name / party_key
    sweep_metadata = {
        "phase": "phase3_sweep",
        **common_metadata,
        "layers": sweep_layers,
        "alphas": sweep_alphas,
        "steering": {
            "injection_mode": "all_positions",
            "vector_normalized": True,
        },
        "vector_metadata_sha256": file_sha256(vector_party_dir / "metadata.json"),
        "vector_sha256_by_layer": {
            str(layer): file_sha256(vector_party_dir / f"vector_layer_{layer:02d}.pt")
            for layer in sweep_layers
        },
    }
    sweep_checkpoint_path = (
        Path(checkpoint_dir) / f"sweep_ckpt_{cfg.short_name}_{party_key}.csv"
    )
    validate_or_create_checkpoint_metadata(
        sweep_checkpoint_path.with_suffix(".metadata.json"),
        sweep_metadata,
        "Phase 3 sweep",
        checkpoint_exists=sweep_checkpoint_path.exists(),
    )

    judge_client = create_judge_client(judge_cfg)
    alpha0_cache_path = (
        Path(checkpoint_dir) / f"alpha0_ckpt_{cfg.short_name}_{party_key}.csv"
    )
    alpha0_df = load_or_build_alpha0_cache(
        alpha0_cache_path,
        {"phase": "phase3_alpha0", **common_metadata},
        model,
        tokenizer,
        cfg,
        sweep_df,
        reasoning_lookup,
        judge_client,
        judge_cfg,
        all_party_keys,
        batch_size,
        sweep_mnt,
        party_key,
    )

    results_df = run_sweep(
        model,
        tokenizer,
        cfg,
        layers_module,
        sweep_df,
        party_key,
        vectors,
        sweep_layers,
        sweep_alphas,
        judge_client,
        judge_cfg,
        reasoning_lookup,
        all_party_keys,
        batch_size,
        Path(checkpoint_dir),
        sweep_mnt,
        alpha0_df,
        sweep_metadata,
    )

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg.short_name
    atomic_write_csv(out_dir / f"sweep_results_{tag}_{party_key}.csv", results_df)
    atomic_write_csv(out_dir / f"alpha0_baseline_{tag}_{party_key}.csv", alpha0_df)
    fingerprint = validate_or_create_checkpoint_metadata(
        Path(checkpoint_dir)
        / f"sweep_ckpt_{cfg.short_name}_{party_key}.metadata.json",
        sweep_metadata,
        "Phase 3 sweep",
        checkpoint_exists=True,
    )["fingerprint"]
    best = refresh_derived_steering_artifacts(
        out_dir,
        tag,
        party_key,
        all_party_keys,
        results_df=results_df,
        alpha0_df=alpha0_df,
        split_manifest=split_manifest,
        phase3_fingerprint=fingerprint,
    )
    print(
        f"\n  Selected on selection set: layer {best['layer']}, "
        f"alpha {best['alpha']}, delta {best['selection_mean_delta']:+.3f}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
