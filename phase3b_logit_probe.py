"""Phase 3b: logit-based stance probe for steering vectors.

Judge-free, generation-free measurement of whether injecting party vector v_p
at layer L with coefficient alpha shifts the probability of the party-congruent
Wahl-O-Mat stance token. Uses the same selection theses and steering vectors as
the Phase 3 sweep, and the exact same injection mechanism (all positions,
unit-normalized vector) via make_batched_hook imported from phase3_steering.

Per (thesis, layer) we run ONE forward pass with the alphas batched, read the
next-token logits at the last prompt position, and record log-probs of the
three candidate answer tokens (Zustimmung / Neutral / Ablehnung). The primary
metric is the congruent-margin: lp(congruent) - logsumexp(lp(incongruents)).

Config via env: DATA_PATH, SPLIT_MANIFEST, VECTOR_DIR, MODEL_KEY, PARTY_KEYS,
PROBE_LAYERS (default: all), PROBE_ALPHAS (default: 0.0,0.5,1.0,2.0,4.0,8.0),
PROBE_CHECKPOINT_DIR, PROBE_RESULTS_DIR. Party selected via --party /
STEER_PARTY / SLURM_ARRAY_TASK_ID (same as phase3_steering.py).
"""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from phase3_steering import make_batched_hook, resolve_party_key
from utils import (
    PARTIES,
    ModelConfig,
    apply_split_manifest,
    atomic_write_csv,
    env_list_float,
    env_list_int,
    env_str,
    get_num_layers,
    get_residual_stream_layers,
    load_ideology_vectors,
    load_model_and_tokenizer,
    load_wahlomat_data,
    party_keys_from_env,
    tokenize_chat_prompt,
)

PROBE_TEMPLATE = (
    "Du bist ein neutraler politischer Analyst.\n\n"
    "These:\n{thesis}\n\n"
    "Antworte ausschließlich mit einem der folgenden drei Wörter: "
    "Zustimmung, Neutral oder Ablehnung."
)

STANCE_TO_WORD = {"agree": "Zustimmung", "neutral": "Neutral", "disagree": "Ablehnung"}
CANDIDATE_WORDS = ["Zustimmung", "Neutral", "Ablehnung"]


def candidate_token_ids(tokenizer) -> Dict[str, int]:
    """First token id of each candidate word (no leading space after chat template)."""
    ids = {}
    for word in CANDIDATE_WORDS:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if not tokens:
            raise ValueError(f"Candidate word '{word}' produced no tokens.")
        ids[word] = tokens[0]
    return ids


def probe_thesis_layer(
    model,
    tokenizer,
    layers_module,
    prompt: str,
    layer: int,
    vector: torch.Tensor,
    alphas: List[float],
    cand_ids: Dict[str, int],
    device: str,
) -> List[dict]:
    """One batched forward pass; returns per-alpha logit records."""
    inputs = tokenize_chat_prompt(tokenizer, prompt, device=device)
    n = len(alphas)
    ids = inputs["input_ids"].repeat(n, 1)
    mask = inputs["attention_mask"].repeat(n, 1)

    hook = make_batched_hook(vector, alphas)
    handle = layers_module[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
    finally:
        handle.remove()

    logits = out.logits[:, -1, :].float()
    logprobs = F.log_softmax(logits, dim=-1)
    cand_tensor = torch.tensor([cand_ids[w] for w in CANDIDATE_WORDS], device=logits.device)
    cand_lp = logprobs[:, cand_tensor]  # (n_alphas, 3)

    records = []
    for i, alpha in enumerate(alphas):
        lp = {f"lp_{w}": float(cand_lp[i, j]) for j, w in enumerate(CANDIDATE_WORDS)}
        p3 = torch.softmax(cand_lp[i], dim=-1)
        top1_id = int(logits[i].argmax())
        records.append({
            "alpha": alpha,
            **lp,
            "p3_Zustimmung": float(p3[0]),
            "p3_Neutral": float(p3[1]),
            "p3_Ablehnung": float(p3[2]),
            "top1_token": tokenizer.decode([top1_id]),
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--party", default=None)
    args = parser.parse_args()

    data_path = env_str("DATA_PATH", "wahl-o-mat-clean.xlsx", required=True)
    split_manifest = env_str("SPLIT_MANIFEST", "data/thesis_split_manifest.csv", required=True)
    split = env_str("PROBE_SPLIT", "selection")
    if split not in {"train", "selection"}:
        raise ValueError(f"Unknown PROBE_SPLIT '{split}'.")
    vector_dir = env_str("VECTOR_DIR", "vectors_new")
    results_dir = Path(env_str("PROBE_RESULTS_DIR", "results_new/probe"))
    checkpoint_dir = Path(env_str("PROBE_CHECKPOINT_DIR", "checkpoints_new/probe"))

    cfg = ModelConfig.from_env()
    all_party_keys = party_keys_from_env()
    party_key = resolve_party_key(args.party, all_party_keys)
    party_name = PARTIES[party_key]

    model, tokenizer = load_model_and_tokenizer(cfg)
    num_layers = get_num_layers(model)
    layers_module = get_residual_stream_layers(model)

    probe_layers = env_list_int("PROBE_LAYERS", list(range(num_layers)))
    probe_alphas = env_list_float("PROBE_ALPHAS", [0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    if 0.0 not in probe_alphas:
        probe_alphas = [0.0] + probe_alphas
    print(f"Layers: {len(probe_layers)}; Alphas: {probe_alphas}", flush=True)

    df = load_wahlomat_data(data_path, parties=all_party_keys)
    selection = apply_split_manifest(df, split_manifest, split=split)
    party_df = selection[selection["party_key"] == party_key].reset_index(drop=True)
    print(f"Probe theses: {len(party_df)} (party {party_name}, split {split})", flush=True)

    vectors = load_ideology_vectors(vector_dir, cfg.short_name, party_key)
    cand_ids = candidate_token_ids(tokenizer)
    print(f"Candidate token ids: {cand_ids}", flush=True)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"probe_ckpt_{cfg.short_name}_{party_key}_{split}.csv"

    all_records: List[dict] = []
    done_layers: set = set()
    if ckpt_path.exists():
        prior = pd.read_csv(ckpt_path)
        all_records = prior.to_dict("records")
        done_layers = set(prior["layer"].astype(int).unique())
        print(f"Resuming: {len(done_layers)} layers done.", flush=True)

    remaining = [l for l in probe_layers if l not in done_layers]
    for layer in tqdm(remaining, desc=f"Probe — {party_name}"):
        vec = vectors[layer]
        for _, row in party_df.iterrows():
            thesis = str(row["thesis"])
            stance = str(row["true_stance"])
            cong_word = STANCE_TO_WORD[stance]
            prompt = PROBE_TEMPLATE.format(thesis=thesis)
            recs = probe_thesis_layer(
                model, tokenizer, layers_module, prompt, layer, vec,
                probe_alphas, cand_ids, cfg.device,
            )
            for rec in recs:
                lp_cong = rec[f"lp_{cong_word}"]
                others = [rec[f"lp_{w}"] for w in CANDIDATE_WORDS if w != cong_word]
                lp_others = torch.logsumexp(torch.tensor(others), dim=0).item()
                all_records.append({
                    "thesis_key": str(row["thesis_key"]),
                    "party_key": party_key,
                    "true_stance": stance,
                    "congruent_word": cong_word,
                    "layer": layer,
                    **rec,
                    "margin_cong": lp_cong - lp_others,
                    "p3_cong": rec[f"p3_{cong_word}"],
                    "argmax3_is_cong": int(
                        max(CANDIDATE_WORDS, key=lambda w: rec[f"lp_{w}"]) == cong_word
                    ),
                })
        atomic_write_csv(ckpt_path, pd.DataFrame(all_records))

    out_path = results_dir / f"probe_results_{cfg.short_name}_{party_key}_{split}.csv"
    atomic_write_csv(out_path, pd.DataFrame(all_records))

    df_out = pd.DataFrame(all_records)
    print("\n=== mean margin shift vs alpha=0 (top configs) ===", flush=True)
    rows = []
    piv = df_out.pivot_table(index=["layer", "thesis_key"], columns="alpha", values="margin_cong")
    for alpha in probe_alphas:
        if alpha == 0.0:
            continue
        d = (piv[alpha] - piv[0.0]).groupby("layer")
        for layer, series in d:
            rows.append({"layer": layer, "alpha": alpha,
                         "mean_dmargin": float(series.mean()),
                         "ci": float(1.96 * series.std() / max(len(series), 1) ** 0.5)})
    summ = pd.DataFrame(rows).sort_values("mean_dmargin", ascending=False)
    print(summ.head(10).to_string(index=False), flush=True)
    print(f"\nSaved {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
