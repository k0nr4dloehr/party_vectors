"""Phase 2 — Similarity-weighted ideology vector extraction.

Vectors are prompt-last-token contrasts against the other-party mean:
for each thesis, all party prompts share the identical template, so
register/persona cancels and only party-specific content survives:

    v_P = weighted_mean_s[ h(prompt_P(s)) - mean_{Q != P} h(prompt_Q(s)) ]

By construction the per-layer vectors sum to zero across parties
(self-centering). Phase 1 responses are only used for similarity weights.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    PARTIES,
    ModelConfig,
    build_party_prompt,
    env_str,
    extract_prompt_last_token_activations,
    get_num_layers,
    load_model_and_tokenizer,
    load_phase1_results,
    party_keys_from_env,
    save_ideology_vectors,
    similarity_weight,
)


def _meta_value(value: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-serializable Python types."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def compute_other_parties_vectors(
    model,
    tokenizer,
    cfg,
    num_layers: int,
    party_keys: List[str],
    phase1_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, tuple[Dict[int, torch.Tensor], float, List[Dict[str, Any]], Dict[str, Any]]]:
    """Contrast each party against the mean of the other parties' prompts."""
    indexed = {
        p: df.set_index(df["thesis_key"].astype(str), drop=False)
        for p, df in phase1_dfs.items()
    }
    key_sets = [set(idx.index) for idx in indexed.values()]
    common_keys = set.intersection(*key_sets)
    if any(len(keys) != len(common_keys) for keys in key_sets):
        print(
            f"  Warning: thesis_key sets differ across parties; "
            f"using {len(common_keys)} shared theses.",
            flush=True,
        )

    n_parties = len(party_keys)
    diff_sum: Dict[str, Dict[int, torch.Tensor]] = {p: {} for p in party_keys}
    weight_sum: Dict[str, float] = {p: 0.0 for p in party_keys}
    statement_weights: Dict[str, List[Dict[str, Any]]] = {p: [] for p in party_keys}
    used_count: Dict[str, int] = {p: 0 for p in party_keys}
    score_values: Dict[str, List[float]] = {p: [] for p in party_keys}

    for thesis_key in tqdm(sorted(common_keys), desc="  theses", leave=False):
        rows = {p: indexed[p].loc[thesis_key] for p in party_keys}
        if any(isinstance(r, pd.DataFrame) for r in rows.values()):
            raise ValueError(f"Duplicate thesis_key in phase 1 results: {thesis_key}")
        thesis = str(rows[party_keys[0]]["thesis"] or "")

        acts = {
            p: extract_prompt_last_token_activations(
                model, tokenizer, build_party_prompt(PARTIES[p], thesis), cfg, num_layers
            )
            for p in party_keys
        }
        layer_sums = {
            layer: torch.stack([acts[p][layer] for p in party_keys]).sum(dim=0)
            for layer in range(num_layers)
        }

        for p in party_keys:
            row = rows[p]
            weight = similarity_weight(row.get("similarity_score"))
            score = row.get("similarity_score")
            if score is not None and not pd.isna(score):
                score_values[p].append(float(score))
            base_meta = {
                "statement_idx": _meta_value(row.get("statement_idx")),
                "election_id": _meta_value(row.get("election_id")),
                "thesis_nr": _meta_value(row.get("thesis_nr")),
                "thesis_key": _meta_value(row.get("thesis_key")),
                "party_statement_key": _meta_value(row.get("party_statement_key")),
                "similarity_score": _meta_value(score),
                "weight": weight,
                "used": False,
            }
            if weight <= 0:
                statement_weights[p].append(base_meta)
                continue

            for layer in range(num_layers):
                other_mean = (layer_sums[layer] - acts[p][layer]) / (n_parties - 1)
                weighted_diff = weight * (acts[p][layer] - other_mean)
                diff_sum[p][layer] = (
                    weighted_diff
                    if layer not in diff_sum[p]
                    else diff_sum[p][layer] + weighted_diff
                )
            weight_sum[p] += weight
            used_count[p] += 1
            statement_weights[p].append({**base_meta, "used": True})

    results = {}
    for p in party_keys:
        if weight_sum[p] <= 0:
            raise RuntimeError(
                f"No usable weighted statements for {PARTIES[p]}. "
                "Check Phase 1 similarity scores."
            )
        vectors = {layer: diff_sum[p][layer] / weight_sum[p] for layer in range(num_layers)}
        scores = score_values[p]
        diagnostics = {
            "used_count": used_count[p],
            "total_rows": len(phase1_dfs[p]),
            "contrast": "party_minus_other_parties_mean",
            "score_mean": float(sum(scores) / len(scores)) if scores else None,
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
        }
        print(
            f"  {PARTIES[p]}: used {used_count[p]}/{len(phase1_dfs[p])} statements "
            f"(sum of weights={weight_sum[p]:.2f}).",
            flush=True,
        )
        results[p] = (vectors, weight_sum[p], statement_weights[p], diagnostics)
    return results


def main() -> int:
    vector_dir = env_str("VECTOR_DIR", "vectors_new")
    baseline_dir = env_str("BASELINE_RESULTS_DIR", "results_new")

    cfg = ModelConfig.from_env()
    party_keys = party_keys_from_env()

    print(f"Model:    {cfg.model_name} ({cfg.short_name})", flush=True)
    print(f"Parties:  {party_keys}", flush=True)
    print(f"Baseline: {baseline_dir}", flush=True)

    model, tokenizer = load_model_and_tokenizer(cfg)
    num_layers = get_num_layers(model)
    print(f"Layers:   {num_layers}", flush=True)

    baseline_path = Path(baseline_dir) / f"baseline_results_{cfg.short_name}.csv"
    phase1_dfs = {
        p: load_phase1_results(baseline_dir, cfg.short_name, p) for p in party_keys
    }

    print("\n=== Computing other-parties-contrast vectors ===", flush=True)
    results = compute_other_parties_vectors(
        model, tokenizer, cfg, num_layers, party_keys, phase1_dfs
    )

    for party_key in party_keys:
        vectors, weight_sum, statement_weights, diagnostics = results[party_key]
        out = save_ideology_vectors(
            vectors,
            vector_dir,
            cfg.short_name,
            party_key,
            cfg.model_name,
            diagnostics["used_count"],
            weight_sum,
            statement_weights,
            source_results_path=str(baseline_path),
            weighting_diagnostics=diagnostics,
        )
        print(f"  Saved {num_layers} vectors for {PARTIES[party_key]} -> {out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
