"""Build a representative human-annotation sample of LLM-as-judge items.

The Phase-3 judge scored generated reasonings against each party's original
Wahl-O-Mat Begründung on a 1–10 similarity scale. This script samples that
judged universe so human annotators can repeat the same scoring task.

Eligible pool
-------------
- Unsteered α=0 responses from ``data/results_new/steering/alpha0_baseline_*``.
- Best-config steered responses from ``sweep_results_*`` at the
  ``selected_config_*`` (layer, alpha).
- Optional Phase-1 party-prompted responses (train theses) so all 114 unique
  theses can appear. Phase 3 only covers the 57 selection theses.

Sampling (defaults)
-------------------
- At most 130 unique theses (there are 114; all are kept).
- Every model, every party, both unsteered and steered, every election, and
  every major topic appear.
- One compared party per item (same isolated-judge protocol).
- LLM scores are written only to a gold file; annotators do not see them.

Usage
-----
    python sample_human_annotation.py
    python sample_human_annotation.py --no-phase1 --max-theses 57
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PARTIES: Dict[str, str] = {
    "CDU_CSU": "CDU / CSU",
    "GRUENE": "GRÜNE",
    "SPD": "SPD",
    "AfD": "AfD",
    "DIE_LINKE": "DIE LINKE",
}
PARTY_KEYS: Tuple[str, ...] = tuple(PARTIES.keys())
# Longest-first so CDU_CSU / DIE_LINKE are not parsed as SPD / AfD suffixes.
PARTY_KEYS_PARSE: Tuple[str, ...] = tuple(
    sorted(PARTY_KEYS, key=len, reverse=True)
)

MODEL_LABELS: Dict[str, str] = {
    "meta-llama-3-8b-instruct": "Llama-3-8B-Instruct",
    "meta-llama-3-70b-instruct": "Llama-3-70B-Instruct",
    "llama-3-2-1b-instruct": "Llama-3.2-1B-Instruct",
    "llama-3-2-3b-instruct": "Llama-3.2-3B-Instruct",
    "qwen3-8b": "Qwen3-8B",
    "ministral-8b-instruct-2410": "Ministral-8B-Instruct",
    "deepseek-llm-7b-chat": "DeepSeek-LLM-7B-Chat",
    "gemma-4-e4b-it": "Gemma-4-E4B-IT",
}

CONDITION_UNSTEERED = "unsteered"
CONDITION_STEERED = "steered"
CONDITION_PARTY_PROMPTED = "party_prompted"

SCORE_BINS: Tuple[str, ...] = ("low_1_3", "mid_4_7", "high_8_10")

ANNOTATOR_COLUMNS: Tuple[str, ...] = (
    "item_id",
    "these",
    "partei",
    "originale_parteiaussage",
    "generierte_antwort",
    "human_score",
    "human_comment",
)

INSTRUCTIONS_DE = (
    "Aufgabe\n"
    "-------\n"
    "Bewerten Sie, wie ähnlich die generierte Antwort in Position und "
    "Argumentation der originalen Parteiaussage ist.\n\n"
    "Skala (ganze Zahlen)\n"
    "--------------------\n"
    "1  = völlig unterschiedliche Position und Argumentation\n"
    "10 = nahezu identische Position und Argumentation\n\n"
    "Vorgehen\n"
    "--------\n"
    "1. Lesen Sie die These, dann die originale Parteiaussage, dann die "
    "generierte Antwort.\n"
    "2. Tragen Sie in der Spalte human_score eine ganze Zahl von 1 bis 10 ein.\n"
    "3. Optional: kurze Notiz in human_comment.\n\n"
    "Bitte nicht recherchieren, nicht mit anderen Annotatorinnen oder "
    "Annotatoren absprechen, und die vorgegebene Reihenfolge nicht ändern. "
    "Die Zeilen sind absichtlich gemischt (verschiedene Modelle, Parteien und "
    "Bedingungen).\n"
)


@dataclass(frozen=True)
class Paths:
    root: Path
    steering: Path
    baseline_dir: Path
    manifest: Path
    workbook: Path
    out_dir: Path


@dataclass(frozen=True)
class SampleConfig:
    max_theses: int = 130
    seed: int = 42
    include_phase1: bool = True


def parse_model_party(filename: str, prefix: str) -> Tuple[str, str]:
    """Split ``<prefix><model>_<PARTY_KEY>`` into ``(model, party_key)``."""
    stem = Path(filename).stem
    if not stem.startswith(prefix):
        raise ValueError(f"Expected prefix {prefix!r} in {filename}")
    rest = stem[len(prefix) :]
    for party_key in PARTY_KEYS_PARSE:
        suffix = f"_{party_key}"
        if rest.endswith(suffix):
            model = rest[: -len(suffix)]
            if not model:
                raise ValueError(f"Empty model name in {filename}")
            return model, party_key
    raise ValueError(f"Cannot parse model/party from {filename}")


def score_bin(score: Any) -> str:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "missing"
    value = float(score)
    if value <= 3:
        return "low_1_3"
    if value <= 7:
        return "mid_4_7"
    return "high_8_10"


def _normalize_party_token(name: str) -> str:
    text = str(name).strip().upper()
    replacements = {
        "Ä": "AE",
        "Ö": "OE",
        "Ü": "UE",
        "ß": "SS",
        "/": " ",
        "-": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    return text


def party_key_from_name(name: Any) -> Optional[str]:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    raw = str(name).strip()
    for key, label in PARTIES.items():
        if raw == label or raw == key:
            return key
    normalized = _normalize_party_token(raw)
    aliases = {
        "CDU CSU": "CDU_CSU",
        "CDUCSU": "CDU_CSU",
        "GRUENE": "GRUENE",
        "GRUNE": "GRUENE",
        "BUENDNIS 90 DIE GRUENEN": "GRUENE",
        "SPD": "SPD",
        "AFD": "AfD",
        "DIE LINKE": "DIE_LINKE",
        "LINKE": "DIE_LINKE",
    }
    return aliases.get(normalized)


def _find_column(columns: Sequence[str], *needles: str) -> str:
    lowered = {c: c.lower() for c in columns}
    for needle in needles:
        needle_l = needle.lower()
        for original, low in lowered.items():
            if needle_l in low:
                return original
    raise ValueError(
        f"Could not find a column matching {needles!r}. Available: {list(columns)}"
    )


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = [
        "thesis_key",
        "election_id",
        "thesis_nr",
        "major_topic",
        "topic_group",
        "split",
        "thesis",
    ]
    missing = [c for c in required if c not in manifest.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    return manifest.drop_duplicates("thesis_key").copy()


def load_party_reasonings(workbook: Path) -> pd.DataFrame:
    """Return one row per (thesis_key, party_key) with original Begründung."""
    raw = pd.read_excel(workbook)
    election_col = _find_column(raw.columns, "Wahl", "election_id")
    thesis_nr_col = _find_column(raw.columns, "These: Nr.")
    thesis_col = _find_column(raw.columns, "These: These")
    reasoning_col = _find_column(raw.columns, "Position: Begründung", "Position: Begr")
    party_col = (
        "Partei: Kurzbezeichnung"
        if "Partei: Kurzbezeichnung" in raw.columns
        else _find_column(raw.columns, "Partei: Name", "Partei")
    )
    frame = pd.DataFrame(
        {
            "election_id": raw[election_col].astype(str).str.strip(),
            "thesis_nr": raw[thesis_nr_col],
            "thesis": raw[thesis_col].astype(str).str.strip(),
            "party_reasoning": raw[reasoning_col].astype(str),
            "party_key": raw[party_col].map(party_key_from_name),
        }
    )
    frame = frame.dropna(subset=["party_key"]).copy()
    frame["thesis_key"] = (
        frame["election_id"].astype(str) + "::" + frame["thesis_nr"].astype(str)
    )
    frame = frame[frame["party_key"].isin(PARTY_KEYS)]
    frame = frame.drop_duplicates(["thesis_key", "party_key"], keep="first")
    return frame[["thesis_key", "party_key", "party_reasoning", "thesis"]]


def _sim_columns(frame: pd.DataFrame) -> List[str]:
    return [f"sim_{pk}" for pk in PARTY_KEYS if f"sim_{pk}" in frame.columns]


def _expand_compared_parties(
    frame: pd.DataFrame,
    *,
    response_col: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sim_cols = _sim_columns(frame)
    if not sim_cols:
        raise ValueError("No sim_<party> columns found in judged results.")
    records = frame.to_dict("records")
    for record in records:
        response = record.get(response_col, "")
        if response is None or (isinstance(response, float) and pd.isna(response)):
            continue
        response_text = str(response).strip()
        if not response_text:
            continue
        for party_key in PARTY_KEYS:
            sim_col = f"sim_{party_key}"
            if sim_col not in record:
                continue
            score = record[sim_col]
            if score is None or (isinstance(score, float) and pd.isna(score)):
                continue
            rows.append(
                {
                    "model": record["model"],
                    "condition": record["condition"],
                    "thesis_key": record["thesis_key"],
                    "thesis": record.get("thesis", ""),
                    "split": record.get("split", ""),
                    "steered_party_key": record.get("steered_party_key"),
                    "prompted_party_key": record.get("prompted_party_key"),
                    "layer": record.get("layer"),
                    "alpha": record.get("alpha"),
                    "generated_response": response_text,
                    "compared_party_key": party_key,
                    "llm_score": float(score),
                    "llm_rationale": str(
                        record.get(f"judge_rationale_{party_key}", "") or ""
                    ),
                    "score_bin": score_bin(score),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def discover_steering_pairs(steering_dir: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for path in sorted(steering_dir.glob("sweep_results_*.csv")):
        model, party_key = parse_model_party(path.name, "sweep_results_")
        pairs.append((model, party_key))
    if not pairs:
        raise FileNotFoundError(f"No sweep_results_*.csv files in {steering_dir}")
    return pairs


def _load_selected_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_unsteered_pool(steering_dir: Path) -> pd.DataFrame:
    """One α=0 response per (model, thesis); reused across steered parties."""
    frames: List[pd.DataFrame] = []
    for path in sorted(steering_dir.glob("alpha0_baseline_*.csv")):
        model, party_key = parse_model_party(path.name, "alpha0_baseline_")
        frame = pd.read_csv(path)
        if "model" not in frame.columns:
            frame["model"] = model
        else:
            frame["model"] = model
        frame["condition"] = CONDITION_UNSTEERED
        frame["steered_party_key"] = None
        frame["prompted_party_key"] = None
        frame["layer"] = np.nan
        frame["alpha"] = 0.0
        frame["_source_party"] = party_key
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No alpha0_baseline_*.csv files in {steering_dir}")
    combined = pd.concat(frames, ignore_index=True)
    # α=0 is methodology-independent and should be identical across party files.
    combined = combined.sort_values(["model", "thesis_key", "_source_party"])
    deduped = combined.drop_duplicates(["model", "thesis_key"], keep="first")
    expanded = _expand_compared_parties(deduped, response_col="steered_response")
    return expanded


def load_steered_pool(steering_dir: Path) -> pd.DataFrame:
    """Best (layer, alpha) steered response per (model, steered party, thesis)."""
    frames: List[pd.DataFrame] = []
    skipped_alpha0 = 0
    for model, party_key in discover_steering_pairs(steering_dir):
        cfg_path = steering_dir / f"selected_config_{model}_{party_key}.json"
        sweep_path = steering_dir / f"sweep_results_{model}_{party_key}.csv"
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        if not sweep_path.exists():
            raise FileNotFoundError(sweep_path)
        config = _load_selected_config(cfg_path)
        layer = int(config["layer"])
        alpha = float(config["alpha"])
        if alpha == 0.0:
            skipped_alpha0 += 1
            continue
        sweep = pd.read_csv(sweep_path)
        selected = sweep[
            (sweep["layer"].astype(int) == layer)
            & (np.isclose(sweep["alpha"].astype(float), alpha))
        ].copy()
        if selected.empty:
            raise ValueError(
                f"No sweep rows for {model} {party_key} at layer={layer} alpha={alpha}"
            )
        selected["model"] = model
        selected["condition"] = CONDITION_STEERED
        selected["steered_party_key"] = party_key
        selected["prompted_party_key"] = None
        frames.append(selected)
    if skipped_alpha0:
        print(
            f"Skipped {skipped_alpha0} selected configs with alpha=0 "
            "(already covered by unsteered items).",
            flush=True,
        )
    if not frames:
        raise ValueError("No steered rows with alpha>0 were found.")
    combined = pd.concat(frames, ignore_index=True)
    return _expand_compared_parties(combined, response_col="steered_response")


def load_phase1_pool(baseline_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(baseline_dir.glob("baseline_results_*.csv")):
        model = path.stem[len("baseline_results_") :]
        frame = pd.read_csv(path)
        frame["model"] = model
        frame["condition"] = CONDITION_PARTY_PROMPTED
        frame["steered_party_key"] = None
        frame["prompted_party_key"] = frame["party_key"]
        frame["layer"] = np.nan
        frame["alpha"] = np.nan
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return _expand_compared_parties(combined, response_col="response_party")


def attach_metadata(
    pool: pd.DataFrame,
    manifest: pd.DataFrame,
    reasonings: pd.DataFrame,
) -> pd.DataFrame:
    meta_cols = [
        "thesis_key",
        "election_id",
        "thesis_nr",
        "major_topic",
        "topic_group",
        "split",
        "title",
        "thesis",
    ]
    present_meta = [c for c in meta_cols if c in manifest.columns]
    merged = pool.merge(
        manifest[present_meta],
        on="thesis_key",
        how="left",
        suffixes=("", "_manifest"),
    )
    if "thesis_manifest" in merged.columns:
        merged["thesis"] = merged["thesis"].where(
            merged["thesis"].astype(str).str.strip() != "",
            merged["thesis_manifest"],
        )
    reasoning_lookup = reasonings.rename(
        columns={"party_key": "compared_party_key"}
    )[["thesis_key", "compared_party_key", "party_reasoning"]]
    merged = merged.merge(
        reasoning_lookup, on=["thesis_key", "compared_party_key"], how="left"
    )
    missing_reason = merged["party_reasoning"].isna().sum()
    if missing_reason:
        raise ValueError(
            f"{missing_reason} pool rows have no original party reasoning. "
            "Check the Wahl-O-Mat workbook join."
        )
    merged["compared_party"] = merged["compared_party_key"].map(PARTIES)
    merged["model_label"] = merged["model"].map(
        lambda key: MODEL_LABELS.get(key, key)
    )
    return merged


def sample_theses(
    manifest: pd.DataFrame,
    available_keys: Iterable[str],
    max_theses: int,
    seed: int,
) -> pd.DataFrame:
    """Keep every available thesis unless the cap is exceeded; then stratify."""
    available = set(available_keys)
    candidates = manifest[manifest["thesis_key"].isin(available)].copy()
    candidates = candidates.drop_duplicates("thesis_key")
    if candidates.empty:
        raise ValueError("No theses in the pool match the split manifest.")
    if len(candidates) <= max_theses:
        return candidates.sort_values(["election_id", "thesis_nr", "thesis_key"])

    rng = np.random.default_rng(seed)
    group_cols = ["election_id", "major_topic", "split"]
    grouped = list(candidates.groupby(group_cols, sort=True))
    n_total = len(candidates)
    # Keep at least one thesis per stratum only when that still fits the cap.
    min_per_group = 1 if len(grouped) <= max_theses else 0
    allocations: Dict[Tuple[Any, ...], int] = {}
    assigned = 0
    for key, group in grouped:
        share = max(min_per_group, int(round(max_theses * len(group) / n_total)))
        share = min(share, len(group))
        allocations[key] = share
        assigned += share
    keys = list(allocations)
    group_sizes = {key: len(group) for key, group in grouped}

    def _decrement_candidates() -> List[Tuple[Any, ...]]:
        if min_per_group:
            return [k for k in keys if allocations[k] > min_per_group]
        return [k for k in keys if allocations[k] > 0]

    while assigned > max_theses:
        candidates_down = _decrement_candidates()
        if not candidates_down:
            # Last resort: drop remaining surplus from any non-empty stratum.
            candidates_down = [k for k in keys if allocations[k] > 0]
        if not candidates_down:
            break
        pick = candidates_down[int(rng.integers(0, len(candidates_down)))]
        allocations[pick] -= 1
        assigned -= 1
    while assigned < max_theses:
        growable = [k for k in keys if allocations[k] < group_sizes[k]]
        if not growable:
            break
        pick = growable[int(rng.integers(0, len(growable)))]
        allocations[pick] += 1
        assigned += 1

    sampled_keys: List[str] = []
    for key, group in grouped:
        n_take = allocations[key]
        if n_take <= 0:
            continue
        chosen = group.sample(n=n_take, random_state=seed)
        sampled_keys.extend(chosen["thesis_key"].tolist())
    sampled = candidates[candidates["thesis_key"].isin(sampled_keys)]
    sampled = sampled.drop_duplicates("thesis_key")
    if len(sampled) > max_theses:
        sampled = sampled.sample(n=max_theses, random_state=seed)
    return sampled.sort_values(["election_id", "thesis_nr", "thesis_key"])


def _filter_rows(frame: pd.DataFrame, **equals: Any) -> pd.DataFrame:
    out = frame
    for column, value in equals.items():
        if value is None:
            continue
        out = out[out[column] == value]
    return out


def _pick_balanced(
    candidates: pd.DataFrame,
    bin_counts: Counter,
    model_counts: Counter,
    rng: np.random.Generator,
) -> Optional[pd.Series]:
    if candidates.empty:
        return None
    ranked = candidates.copy()
    ranked["_bin_rank"] = ranked["score_bin"].map(lambda b: bin_counts[b])
    ranked["_model_rank"] = ranked["model"].map(lambda model: model_counts[model])
    min_bin = ranked["_bin_rank"].min()
    tied_bin = ranked[ranked["_bin_rank"] == min_bin]
    min_model = tied_bin["_model_rank"].min()
    tied = tied_bin[tied_bin["_model_rank"] == min_model]
    index = int(rng.integers(0, len(tied)))
    return tied.iloc[index].drop(labels=["_bin_rank", "_model_rank"])


def _pick_with_relaxation(
    pool: pd.DataFrame,
    filters: Sequence[Dict[str, Any]],
    bin_counts: Counter,
    model_counts: Counter,
    rng: np.random.Generator,
) -> Optional[pd.Series]:
    for constraint in filters:
        match = _filter_rows(pool, **constraint)
        picked = _pick_balanced(match, bin_counts, model_counts, rng)
        if picked is not None:
            return picked
    return None


def select_representative_items(
    pool: pd.DataFrame,
    theses: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """One unsteered + one steered item per selection thesis; one Phase-1 item per train thesis."""
    rng = np.random.default_rng(seed)
    models = sorted(pool["model"].unique().tolist())
    if not models:
        raise ValueError("Pool has no models.")
    selection = theses[theses["split"] == "selection"]["thesis_key"].tolist()
    train = theses[theses["split"] == "train"]["thesis_key"].tolist()
    bin_counts: Counter = Counter()
    model_counts: Counter = Counter()
    picked_rows: List[pd.Series] = []

    def accept(row: Optional[pd.Series], slot: str) -> None:
        if row is None:
            raise ValueError(f"Could not fill annotation slot {slot!r}.")
        item = row.copy()
        item["assignment_slot"] = slot
        picked_rows.append(item)
        bin_counts[item["score_bin"]] += 1
        model_counts[item["model"]] += 1

    for index, thesis_key in enumerate(selection):
        model_u = models[index % len(models)]
        compared_u = PARTY_KEYS[index % len(PARTY_KEYS)]
        unsteered_pool = pool[
            (pool["thesis_key"] == thesis_key)
            & (pool["condition"] == CONDITION_UNSTEERED)
        ]
        accept(
            _pick_with_relaxation(
                unsteered_pool,
                [
                    {"model": model_u, "compared_party_key": compared_u},
                    {"model": model_u},
                    {"compared_party_key": compared_u},
                    {},
                ],
                bin_counts,
                model_counts,
                rng,
            ),
            "unsteered",
        )

        model_s = models[(index + len(models) // 2) % len(models)]
        steered_party = PARTY_KEYS[(index + 2) % len(PARTY_KEYS)]
        compared_s = (
            steered_party
            if index % 2 == 0
            else PARTY_KEYS[(index + 3) % len(PARTY_KEYS)]
        )
        steered_pool = pool[
            (pool["thesis_key"] == thesis_key)
            & (pool["condition"] == CONDITION_STEERED)
        ]
        accept(
            _pick_with_relaxation(
                steered_pool,
                [
                    {
                        "model": model_s,
                        "steered_party_key": steered_party,
                        "compared_party_key": compared_s,
                    },
                    {
                        "model": model_s,
                        "steered_party_key": steered_party,
                    },
                    {"steered_party_key": steered_party},
                    {"model": model_s},
                    {},
                ],
                bin_counts,
                model_counts,
                rng,
            ),
            "steered",
        )

    for index, thesis_key in enumerate(train):
        model_p = models[index % len(models)]
        prompted = PARTY_KEYS[index % len(PARTY_KEYS)]
        compared_p = PARTY_KEYS[(index + 1) % len(PARTY_KEYS)]
        train_pool = pool[
            (pool["thesis_key"] == thesis_key)
            & (pool["condition"] == CONDITION_PARTY_PROMPTED)
        ]
        if train_pool.empty:
            continue
        accept(
            _pick_with_relaxation(
                train_pool,
                [
                    {
                        "model": model_p,
                        "prompted_party_key": prompted,
                        "compared_party_key": compared_p,
                    },
                    {"model": model_p, "prompted_party_key": prompted},
                    {"model": model_p},
                    {},
                ],
                bin_counts,
                model_counts,
                rng,
            ),
            "party_prompted",
        )

    if not picked_rows:
        raise ValueError("Representative sample is empty.")
    return pd.DataFrame(picked_rows).reset_index(drop=True)


def shuffle_and_label(sample: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled.insert(0, "item_id", [f"H-{i:03d}" for i in range(1, len(shuffled) + 1)])
    annotator = pd.DataFrame(
        {
            "item_id": shuffled["item_id"],
            "these": shuffled["thesis"],
            "partei": shuffled["compared_party"],
            "originale_parteiaussage": shuffled["party_reasoning"],
            "generierte_antwort": shuffled["generated_response"],
            "human_score": pd.NA,
            "human_comment": "",
        }
    )
    gold_cols = [
        "item_id",
        "assignment_slot",
        "model",
        "model_label",
        "condition",
        "thesis_key",
        "election_id",
        "thesis_nr",
        "split",
        "major_topic",
        "topic_group",
        "steered_party_key",
        "prompted_party_key",
        "compared_party_key",
        "compared_party",
        "layer",
        "alpha",
        "llm_score",
        "score_bin",
        "llm_rationale",
        "thesis",
        "party_reasoning",
        "generated_response",
    ]
    present = [c for c in gold_cols if c in shuffled.columns]
    gold = shuffled[present].copy()
    return annotator, gold


def coverage_report(gold: pd.DataFrame) -> Dict[str, Any]:
    def counts(column: str) -> Dict[str, int]:
        if column not in gold.columns:
            return {}
        return {
            str(key): int(value)
            for key, value in gold[column].value_counts(dropna=False).items()
        }

    return {
        "n_items": int(len(gold)),
        "n_theses": int(gold["thesis_key"].nunique()),
        "n_models": int(gold["model"].nunique()),
        "theses_by_split": {
            str(key): int(value)
            for key, value in gold.drop_duplicates("thesis_key")["split"]
            .value_counts()
            .items()
        },
        "items_by_split": counts("split"),
        "theses_by_election": {
            str(key): int(value)
            for key, value in gold.drop_duplicates("thesis_key")["election_id"]
            .value_counts()
            .items()
        },
        "items_by_model": counts("model"),
        "items_by_condition": counts("condition"),
        "items_by_compared_party": counts("compared_party_key"),
        "items_by_steered_party": counts("steered_party_key"),
        "items_by_score_bin": counts("score_bin"),
        "items_by_topic": counts("major_topic"),
        "llm_score_mean": float(gold["llm_score"].mean()),
        "llm_score_std": float(gold["llm_score"].std(ddof=0)),
    }


def write_excel(annotator: pd.DataFrame, path: Path) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"anleitung": INSTRUCTIONS_DE.split("\n")}).to_excel(
            writer, sheet_name="Anleitung", index=False, header=False
        )
        annotator.to_excel(writer, sheet_name="Annotation", index=False)
        workbook = writer.book
        guide = workbook["Anleitung"]
        items = workbook["Annotation"]
        guide.column_dimensions["A"].width = 110
        for cell in guide["A"]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.font = Font(name="Calibri", size=12)

        header_fill = PatternFill("solid", fgColor="1F4E79")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        wrap = Alignment(wrap_text=True, vertical="top")
        for cell in items[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        widths = {
            "A": 12,
            "B": 42,
            "C": 16,
            "D": 55,
            "E": 55,
            "F": 16,
            "G": 28,
        }
        for letter, width in widths.items():
            items.column_dimensions[letter].width = width
        for row in items.iter_rows(min_row=2, max_col=7):
            for cell in row:
                cell.alignment = wrap
        items.row_dimensions[1].height = 22
        items.freeze_panes = "A2"
        items.auto_filter.ref = items.dimensions
        score_col = get_column_letter(annotator.columns.get_loc("human_score") + 1)
        validation = DataValidation(
            type="whole",
            operator="between",
            formula1="1",
            formula2="10",
            allow_blank=True,
            showErrorMessage=True,
            errorTitle="Ungültiger Score",
            error="Bitte eine ganze Zahl von 1 bis 10 eintragen.",
        )
        validation.add(f"{score_col}2:{score_col}{len(annotator) + 1}")
        items.add_data_validation(validation)


def write_outputs(
    annotator: pd.DataFrame,
    gold: pd.DataFrame,
    report: Dict[str, Any],
    out_dir: Path,
    config: SampleConfig,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "annotator_csv": out_dir / "annotator_items.csv",
        "gold_csv": out_dir / "gold.csv",
        "report_json": out_dir / "coverage_report.json",
        "annotator_xlsx": out_dir / "annotator_items.xlsx",
    }
    annotator.to_csv(paths["annotator_csv"], index=False, encoding="utf-8-sig")
    gold.to_csv(paths["gold_csv"], index=False, encoding="utf-8-sig")
    payload = {
        "config": {
            "max_theses": config.max_theses,
            "seed": config.seed,
            "include_phase1": config.include_phase1,
        },
        "coverage": report,
    }
    paths["report_json"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_excel(annotator, paths["annotator_xlsx"])
    return paths


def build_sample(paths: Paths, config: SampleConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    manifest = load_manifest(paths.manifest)
    reasonings = load_party_reasonings(paths.workbook)
    print("Loading unsteered alpha=0 responses ...", flush=True)
    unsteered = load_unsteered_pool(paths.steering)
    print(f"  {len(unsteered)} compared-party items", flush=True)
    print("Loading best-config steered responses ...", flush=True)
    steered = load_steered_pool(paths.steering)
    print(f"  {len(steered)} compared-party items", flush=True)
    parts = [unsteered, steered]
    if config.include_phase1:
        print("Loading Phase-1 party-prompted responses ...", flush=True)
        phase1 = load_phase1_pool(paths.baseline_dir)
        print(f"  {len(phase1)} compared-party items", flush=True)
        if phase1.empty:
            print("  Warning: no Phase-1 baseline_results_*.csv files found.", flush=True)
        else:
            parts.append(phase1)
    pool = attach_metadata(pd.concat(parts, ignore_index=True), manifest, reasonings)
    print(
        f"Pool: {len(pool)} items, "
        f"{pool['thesis_key'].nunique()} theses, "
        f"{pool['model'].nunique()} models",
        flush=True,
    )
    theses = sample_theses(
        manifest,
        pool["thesis_key"].unique(),
        config.max_theses,
        config.seed,
    )
    print(
        f"Sampled {len(theses)} theses "
        f"(cap {config.max_theses}; split {theses['split'].value_counts().to_dict()})",
        flush=True,
    )
    selected = select_representative_items(pool, theses, config.seed)
    annotator, gold = shuffle_and_label(selected, config.seed)
    leaked = [c for c in annotator.columns if "llm" in c.lower() or c in {"model", "condition"}]
    if leaked:
        raise RuntimeError(f"Annotator sheet leaked gold columns: {leaked}")
    report = coverage_report(gold)
    return annotator, gold, report


def default_paths(root: Path, out_dir: Optional[Path] = None) -> Paths:
    workbook = root / "wahl-o-mat-clean.xlsx"
    if not workbook.exists():
        alt = root / "data" / "wahl-o-mat-clean.xlsx"
        if alt.exists():
            workbook = alt
    return Paths(
        root=root,
        steering=root / "data" / "results_new" / "steering",
        baseline_dir=root / "data" / "results_new",
        manifest=root / "data" / "thesis_split_manifest.csv",
        workbook=workbook,
        out_dir=out_dir or (root / "data" / "human_annotation"),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample judged steered and baseline responses for human annotation."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root (default: this script's directory).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/human_annotation).",
    )
    parser.add_argument("--max-theses", type=int, default=130)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-phase1",
        action="store_true",
        help="Do not include Phase-1 party-prompted train theses.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    paths = default_paths(args.root, args.out_dir)
    config = SampleConfig(
        max_theses=args.max_theses,
        seed=args.seed,
        include_phase1=not args.no_phase1,
    )
    annotator, gold, report = build_sample(paths, config)
    written = write_outputs(annotator, gold, report, paths.out_dir, config)
    print("\nCoverage", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print("\nWrote:", flush=True)
    for label, path in written.items():
        print(f"  {label}: {path}", flush=True)


if __name__ == "__main__":
    main()
