"""Shared utilities for the similarity-weighted ideology vector pipeline.

Configuration is driven entirely by environment variables (see README.md).
No ``.env`` file is loaded; the cluster is expected to export everything.
"""

from __future__ import annotations

import json
import os
import hashlib
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARTIES: Dict[str, str] = {
    "CDU_CSU": "CDU / CSU",
    "GRUENE": "GRÜNE",
    "SPD": "SPD",
    "AfD": "AfD",
    "DIE_LINKE": "DIE LINKE",
}

STANCE_MAPPING: Dict[str, str] = {
    "stimme zu": "agree",
    "stimme nicht zu": "disagree",
    "neutral": "neutral",
}

MODEL_REGISTRY: Dict[str, str] = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
    "mistral": "mistralai/Ministral-8B-Instruct-2410",
    "deepseek": "deepseek-ai/deepseek-llm-7b-chat",
    "gemma": "google/gemma-4-E4B-it",
}

PARTY_CONDITION_TEMPLATE = (
    "Antworte auf die folgende Wahl-O-Mat-These so, wie die {party} antworten würde. "
    "Formuliere eine Position und Begründung.\n\n"
    "These:\n{thesis}"
)

ANTI_CONDITION_TEMPLATE = (
    "Antworte auf die folgende Wahl-O-Mat-These so, wie ein entschiedener Gegner "
    "der {party} antworten würde. Formuliere eine Position und Begründung.\n\n"
    "These:\n{thesis}"
)

NEUTRAL_PERSONA = (
    "Du bist ein neutraler politischer Analyst. "
    "Du analysierst politische Positionen sachlich und ohne Parteipräferenz."
)

SCHEMA_VERSION: int = 3

ELECTION_ID_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "Wahl",
    "Wahl: Name",
    "Wahl: Kurzbezeichnung",
    "Wahl: Jahr",
    "Wahl-O-Mat: Name",
    "WOM: Name",
    "Dataset",
    "Quelle",
    "Source",
    "election_id",
)

JUDGE_BATCH_SIZE: int = 10
# Retry policy must survive a full 60s rate-limit window: 1s/2s backoffs under a
# shared 60 req/min key produced permanent 429s and NaN judge holes.
MAX_JUDGE_RETRIES: int = int(os.environ.get("JUDGE_MAX_RETRIES", "6"))
JUDGE_BACKOFF_BASE: int = 8
JUDGE_BACKOFF_MAX: float = 90.0
# Minimum seconds between judge API calls per process. With N concurrent array
# tasks the global rate is bounded by N * 60/JUDGE_MIN_INTERVAL req/min.
JUDGE_MIN_INTERVAL: float = float(os.environ.get("JUDGE_MIN_INTERVAL", "0"))
_judge_last_call: float = 0.0


def _pace_judge_call() -> None:
    global _judge_last_call
    if JUDGE_MIN_INTERVAL <= 0:
        return
    wait = JUDGE_MIN_INTERVAL - (time.monotonic() - _judge_last_call)
    if wait > 0:
        time.sleep(wait)
    _judge_last_call = time.monotonic()
DEFAULT_GENERATION_MAX_NEW_TOKENS: int = 512
DEFAULT_THESIS_SPLIT_SEED: int = 42
DEFAULT_SPLIT_MANIFEST: str = "data/thesis_split_manifest.csv"


# ---------------------------------------------------------------------------
# Atomic checkpoint and configuration metadata IO
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_csv(path: Path, dataframe: pd.DataFrame) -> None:
    """Write CSV through a same-directory temporary file and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            dataframe.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return canonical checkpoint metadata with a content fingerprint."""
    canonical_payload = json.loads(json.dumps(payload, sort_keys=True, default=str))
    encoded = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "fingerprint_algorithm": "sha256",
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
        "configuration": canonical_payload,
    }


def validate_or_create_checkpoint_metadata(
    path: Path,
    payload: Dict[str, Any],
    context: str,
    checkpoint_exists: bool,
) -> Dict[str, Any]:
    """Validate checkpoint identity or atomically create its metadata sidecar."""
    expected = checkpoint_metadata(payload)
    path = Path(path)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                actual = json.load(handle)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{context} metadata is not valid JSON: {path}. "
                "Clear the checkpoint and metadata files before rerunning."
            ) from exc
        if not isinstance(actual, dict):
            raise RuntimeError(
                f"{context} metadata has an invalid structure: {path}. "
                "Clear the checkpoint and metadata files before rerunning."
            )
        if (
            actual.get("fingerprint") != expected["fingerprint"]
            or actual.get("configuration") != expected["configuration"]
        ):
            raise RuntimeError(
                f"{context} configuration does not match its checkpoint metadata. "
                f"Clear the checkpoint and metadata files before rerunning: "
                f"{path.parent}"
            )
        return actual
    if checkpoint_exists:
        raise RuntimeError(
            f"{context} checkpoint exists without metadata: {path}. "
            "It cannot be resumed safely. Clear the checkpoint and rerun."
        )
    atomic_write_json(path, expected)
    return expected


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def env_str(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    v = os.environ.get(key, default)
    if required and not v:
        raise RuntimeError(f"Required env var '{key}' is not set.")
    return v


def env_int(key: str, default: Optional[int] = None) -> Optional[int]:
    v = os.environ.get(key)
    return int(v) if v not in (None, "") else default


def env_float(key: str, default: Optional[float] = None) -> Optional[float]:
    v = os.environ.get(key)
    return float(v) if v not in (None, "") else default


def env_list_str(key: str, default: Optional[List[str]] = None) -> List[str]:
    v = os.environ.get(key)
    if v is None or v == "":
        return list(default) if default is not None else []
    return [x.strip() for x in v.split(",") if x.strip()]


def env_list_float(key: str, default: Optional[List[float]] = None) -> List[float]:
    v = os.environ.get(key)
    if v is None or v == "":
        return list(default) if default is not None else []
    return [float(x.strip()) for x in v.split(",") if x.strip()]


def env_list_int(key: str, default: Optional[List[int]] = None) -> List[int]:
    v = os.environ.get(key)
    if v is None or v == "":
        return list(default) if default is not None else []
    if ":" in v and "," not in v:
        parts = [int(p) for p in v.split(":")]
        if len(parts) == 2:
            return list(range(parts[0], parts[1]))
        if len(parts) == 3:
            return list(range(parts[0], parts[1], parts[2]))
    return [int(x.strip()) for x in v.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    token: Optional[str] = None

    def __post_init__(self) -> None:
        if self.token is None:
            self.token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if self.temperature == 0.0:
            self.do_sample = False

    @classmethod
    def from_env(cls) -> "ModelConfig":
        key = env_str("MODEL_KEY", "llama3")
        name = MODEL_REGISTRY.get(key, key)
        generation_tokens = generation_max_tokens_from_env()
        return cls(
            model_name=name,
            max_new_tokens=generation_tokens,
            temperature=env_float("TEMPERATURE", 0.0),
            load_in_8bit=env_str("LOAD_IN_8BIT", "0") == "1",
            load_in_4bit=env_str("LOAD_IN_4BIT", "0") == "1",
        )

    @property
    def short_name(self) -> str:
        name = self.model_name.split("/")[-1].lower()
        for ch in (" ", ".", "/"):
            name = name.replace(ch, "-")
        return name


@dataclass
class JudgeConfig:
    model: str = "openai/gpt-oss-120b"
    base_url: str = "https://llm.scads.ai/v1"
    api_key: Optional[str] = None
    max_new_tokens: int = 128
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("API-KEY") or os.getenv("API_KEY")

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        return cls(
            model=env_str("JUDGE_MODEL", "openai/gpt-oss-120b"),
            base_url=env_str("JUDGE_BASE_URL", "https://llm.scads.ai/v1"),
            max_new_tokens=env_int("JUDGE_MAX_NEW_TOKENS", 1024),
            temperature=env_float("JUDGE_TEMPERATURE", 0.0),
        )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_party_prompt(party_name: str, thesis: str) -> str:
    return PARTY_CONDITION_TEMPLATE.format(party=party_name, thesis=thesis)


def build_anti_prompt(party_name: str, thesis: str) -> str:
    return ANTI_CONDITION_TEMPLATE.format(party=party_name, thesis=thesis)


def build_neutral_statement_prompt(thesis: str) -> str:
    return (
        f"{NEUTRAL_PERSONA}\n\n"
        "Du erhältst eine Wahl-O-Mat-These. Deine Aufgabe ist es, "
        "eine politische Position zu dieser These zu formulieren und zu begründen.\n\n"
        f"These:\n{thesis}\n\n"
        "Formuliere eine Position und Begründung zu dieser These:\n"
    )


def clean_response(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    text = text.replace("Ġ", " ")
    text = text.replace("Ċ", "\n")
    text = text.replace("ĉ", "\t")

    for encoding in ["latin1", "cp1252"]:
        try:
            candidate = text.encode(encoding).decode("utf-8")
            text = candidate
            break
        except Exception:
            pass

    lines = []
    for line in text.splitlines():
        cleaned_line = " ".join(line.split())
        if cleaned_line:
            lines.append(cleaned_line)
    text = "\n".join(lines)

    if len(text) > 500:
        text = text[:500] + "..."

    return text


def build_similarity_judge_prompt(
    thesis: str,
    party_reasoning: str,
    response: str,
    party_name: str,
) -> str:
    response = clean_response(response)
    party_reasoning = clean_response(party_reasoning)
    return (
        "Du bist ein politischer Analyst. Bewerte, wie ähnlich die untenstehende "
        "generierte Antwort in Position und Argumentation der originalen "
        "Parteiaussage ist.\n\n"
        "Gib NUR eine JSON-Antwort im folgenden Format zurück:\n"
        "{\n"
        '  "score": <1-10>,\n'
        '  "rationale": "<kurze Begründung>"\n'
        "}\n\n"
        "Skala:\n"
        "1 = völlig unterschiedliche Position und Argumentation\n"
        "10 = nahezu identische Position und Argumentation\n\n"
        f"These:\n{thesis}\n\n"
        f"Originale Parteiaussage ({party_name}):\n{party_reasoning}\n\n"
        f"Generierte Antwort:\n{response}\n"
    )


def build_similarity_judge_batch_prompt(
    items: List[Tuple[str, str, str, str]],
) -> str:
    header = (
        f"Du bist ein politischer Analyst. Unten stehen {len(items)} "
        "These-Antwort-Paare mit der originalen Parteiaussage.\n"
        "Bewerte für jedes Paar, wie ähnlich die generierte Antwort in Position "
        "und Argumentation der originalen Parteiaussage ist.\n\n"
        "Gib NUR ein JSON-Array zurück, mit einem Objekt pro Paar, "
        "in exakt gleicher Reihenfolge:\n"
        '[{"score": <1-10>, "rationale": "<kurze Begründung>"}, ...]\n\n'
        "Skala:\n"
        "1 = völlig unterschiedliche Position und Argumentation\n"
        "10 = nahezu identische Position und Argumentation\n\n"
    )
    blocks = []
    for i, (thesis, party_name, party_reasoning, response) in enumerate(items, 1):
        blocks.append(
            f"--- Paar {i} ---\n"
            f"These: {thesis}\n"
            f"Partei: {party_name}\n"
            f"Originale Parteiaussage: {clean_response(party_reasoning)}\n"
            f"Generierte Antwort: {clean_response(response)}"
        )
    return header + "\n\n".join(blocks)


def build_cross_party_similarity_judge_prompt(
    thesis: str,
    party_reasonings: Dict[str, str],
    response: str,
    party_keys: List[str],
) -> str:
    response = clean_response(response)
    reasoning_lines = []
    for pk in party_keys:
        party_name = PARTIES[pk]
        reasoning = clean_response(party_reasonings.get(pk, "") or "")
        reasoning_lines.append(f"- {party_name} ({pk}): {reasoning}")

    score_template = ", ".join(f'"{pk}": <1-10>' for pk in party_keys)
    return (
        "Du bist ein politischer Analyst. Bewerte für die untenstehende generierte "
        "Antwort, wie ähnlich sie in Position und Argumentation den originalen "
        "Parteiaussagen ist.\n\n"
        "Gib NUR ein JSON-Objekt im folgenden Format zurück:\n"
        "{\n"
        f'  "scores": {{{score_template}}},\n'
        '  "rationale": "<kurze Begründung>"\n'
        "}\n\n"
        "Skala pro Partei:\n"
        "1 = völlig unterschiedliche Position und Argumentation\n"
        "10 = nahezu identische Position und Argumentation\n\n"
        f"These:\n{thesis}\n\n"
        "Originale Parteiaussagen:\n"
        + "\n".join(reasoning_lines)
        + f"\n\nGenerierte Antwort:\n{response}\n"
    )


def build_cross_party_similarity_batch_prompt(
    items: List[Tuple[str, Dict[str, str], str]],
    party_keys: List[str],
) -> str:
    score_template = ", ".join(f'"{pk}": <1-10>' for pk in party_keys)
    header = (
        f"Du bist ein politischer Analyst. Unten stehen {len(items)} "
        "generierte Antworten zu Wahl-O-Mat-Thesen.\n"
        "Bewerte für jede Antwort, wie ähnlich sie in Position und Argumentation "
        "den originalen Parteiaussagen ist.\n\n"
        "Gib NUR ein JSON-Array zurück, mit einem Objekt pro Antwort, "
        "in exakt gleicher Reihenfolge:\n"
        "[\n"
        "  {\n"
        f'    "scores": {{{score_template}}},\n'
        '    "rationale": "<kurze Begründung>"\n'
        "  },\n"
        "  ...\n"
        "]\n\n"
        "Skala pro Partei:\n"
        "1 = völlig unterschiedliche Position und Argumentation\n"
        "10 = nahezu identische Position und Argumentation\n\n"
    )
    blocks = []
    for i, (thesis, party_reasonings, response) in enumerate(items, 1):
        reasoning_lines = []
        for pk in party_keys:
            party_name = PARTIES[pk]
            reasoning = clean_response(party_reasonings.get(pk, "") or "")
            reasoning_lines.append(f"  - {party_name} ({pk}): {reasoning}")
        blocks.append(
            f"--- Antwort {i} ---\n"
            f"These: {thesis}\n"
            "Originale Parteiaussagen:\n"
            + "\n".join(reasoning_lines)
            + f"\nGenerierte Antwort: {clean_response(response)}"
        )
    return header + "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Model / tokenizer
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: ModelConfig):
    print(f"Loading model {cfg.model_name} on {cfg.device} ...", flush=True)

    tok_kwargs: Dict[str, Any] = {}
    if cfg.token:
        tok_kwargs["token"] = cfg.token

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "dtype": torch.bfloat16 if cfg.device == "cuda" else torch.float32,
        "device_map": "auto" if cfg.device == "cuda" else None,
    }
    if cfg.token:
        model_kwargs["token"] = cfg.token
    if "gemma-2" in cfg.model_name.lower():
        model_kwargs["attn_implementation"] = "eager"
    if cfg.load_in_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
    if cfg.device != "cuda" and not cfg.load_in_8bit and not cfg.load_in_4bit:
        model = model.to(cfg.device)

    model.eval()
    print("Model loaded.", flush=True)
    return model, tokenizer


def get_residual_stream_layers(model) -> nn.ModuleList:
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        return model.decoder.layers
    raise AttributeError(
        f"Cannot locate decoder layers on {type(model).__name__}. "
        "Expected model.model.layers or model.decoder.layers."
    )


def get_num_layers(model) -> int:
    return len(get_residual_stream_layers(model))


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    tpl_kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(messages, **tpl_kwargs, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **tpl_kwargs)


def tokenize_chat_prompt(tokenizer, content: str, device: str = "cuda", max_length: int = 2048):
    messages = [{"role": "user", "content": content}]
    formatted = _apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in inputs.items()}


def tokenize_chat_with_response(
    tokenizer,
    user_content: str,
    assistant_content: str,
    device: str = "cuda",
    max_length: int = 4096,
) -> Tuple[Dict[str, torch.Tensor], int]:
    user_messages = [{"role": "user", "content": user_content}]
    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    user_formatted = _apply_chat_template(tokenizer, user_messages, add_generation_prompt=True)
    full_formatted = _apply_chat_template(tokenizer, full_messages, add_generation_prompt=False)

    user_inputs = tokenizer(user_formatted, return_tensors="pt", truncation=True, max_length=max_length)
    full_inputs = tokenizer(full_formatted, return_tensors="pt", truncation=True, max_length=max_length)
    prompt_len = user_inputs["input_ids"].shape[1]
    return {k: v.to(device) for k, v in full_inputs.items()}, prompt_len


def generate_response(
    model,
    tokenizer,
    prompt: str,
    cfg: ModelConfig,
    max_new_tokens: Optional[int] = None,
) -> str:
    mnt = max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens
    inputs = tokenize_chat_prompt(tokenizer, prompt, device=cfg.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=mnt,
            temperature=cfg.temperature if cfg.do_sample else None,
            do_sample=cfg.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_length = inputs["input_ids"].shape[1]
    generated = outputs[0][input_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def extract_response_mean_activations(
    model,
    tokenizer,
    user_prompt: str,
    response: str,
    cfg: ModelConfig,
    num_layers: int,
) -> Dict[int, torch.Tensor]:
    inputs, prompt_len = tokenize_chat_with_response(
        tokenizer, user_prompt, response, device=cfg.device
    )
    seq_len = inputs["input_ids"].shape[1]
    if prompt_len >= seq_len:
        raise ValueError(
            f"No response tokens found (prompt_len={prompt_len}, seq_len={seq_len})."
        )

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states

    activations: Dict[int, torch.Tensor] = {}
    for layer in range(num_layers):
        layer_hs = hs[layer + 1][0, prompt_len:seq_len, :].float()
        activations[layer] = layer_hs.mean(dim=0).cpu()
    return activations


def extract_prompt_last_token_activations(
    model,
    tokenizer,
    user_prompt: str,
    cfg: ModelConfig,
    num_layers: int,
) -> Dict[int, torch.Tensor]:
    """Hidden state at the last prompt position (predicts first response token).

    Uses the generation-format chat template so the extraction site matches
    the site seen during steered generation.
    """
    inputs = tokenize_chat_prompt(tokenizer, user_prompt, device=cfg.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states
    return {
        layer: hs[layer + 1][0, -1, :].float().cpu()
        for layer in range(num_layers)
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generation_max_tokens_from_env() -> int:
    """Shared generation budget for party, neutral, and steering runs."""
    return env_int(
        "GENERATION_MAX_NEW_TOKENS",
        env_int("MAX_NEW_TOKENS", DEFAULT_GENERATION_MAX_NEW_TOKENS),
    )


def detect_election_id_column(df: pd.DataFrame) -> str:
    override = env_str("ELECTION_ID_COLUMN")
    if override:
        if override not in df.columns:
            raise ValueError(
                f"ELECTION_ID_COLUMN='{override}' not found. "
                f"Available: {df.columns.tolist()}"
            )
        return override
    for col in ELECTION_ID_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "No election/source column found. Set ELECTION_ID_COLUMN or add one of "
        f"{ELECTION_ID_COLUMN_CANDIDATES}. Available: {df.columns.tolist()}"
    )


def _party_name_to_key(party_name: Any) -> Optional[str]:
    if party_name is None or (isinstance(party_name, float) and pd.isna(party_name)):
        return None
    name = str(party_name).strip()
    for key, label in PARTIES.items():
        if name == label:
            return key
    return None


def add_canonical_keys(df: pd.DataFrame, party_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """Add election-aware thesis and party-statement keys with validation."""
    out = df.copy()
    if "party_reasoning" not in out.columns and "Position: Begründung" in out.columns:
        out["party_reasoning"] = out["Position: Begründung"].astype(str)
    election_col = detect_election_id_column(out)
    out["election_id"] = out[election_col].astype(str).str.strip()
    out["thesis_nr"] = out["These: Nr."]
    out["thesis"] = out["These: These"].astype(str).str.strip()
    out["thesis_key"] = out["election_id"] + "::" + out["thesis_nr"].astype(str)

    party_col = (
        "Partei: Kurzbezeichnung"
        if "Partei: Kurzbezeichnung" in out.columns
        else "Partei: Name"
    )
    out["party_key"] = out[party_col].map(_party_name_to_key)
    if out["party_key"].isna().any():
        bad = out.loc[out["party_key"].isna(), party_col].unique().tolist()
        raise ValueError(f"Unknown party names in dataset: {bad}")

    if party_keys is not None:
        allowed = set(party_keys)
        out = out[out["party_key"].isin(allowed)].copy()

    out["party_statement_key"] = (
        out["election_id"]
        + "::"
        + out["thesis_nr"].astype(str)
        + "::"
        + out["party_key"].astype(str)
    )
    out["schema_version"] = SCHEMA_VERSION

    duplicate_keys = out["party_statement_key"].duplicated(keep=False)
    if duplicate_keys.any():
        bad_keys = sorted(out.loc[duplicate_keys, "party_statement_key"].unique())[:5]
        raise ValueError(
            "Duplicate party-statement records remain after exact-duplicate removal. "
            f"Examples: {bad_keys}. The source workbook must not contain conflicting "
            "records for one election, thesis, and party."
        )

    thesis_text_counts = out.groupby("thesis_key")["thesis"].nunique(dropna=False)
    if (thesis_text_counts > 1).any():
        bad_keys = thesis_text_counts[thesis_text_counts > 1].index.tolist()[:5]
        raise ValueError(
            "thesis_key maps to multiple thesis texts. Examples: "
            f"{bad_keys}. Check election_id and thesis_nr."
        )

    reasoning_counts = out.groupby("party_statement_key")["party_reasoning"].nunique(dropna=False)
    if (reasoning_counts > 1).any():
        bad_keys = reasoning_counts[reasoning_counts > 1].index.tolist()[:5]
        raise ValueError(
            "party_statement_key maps to multiple reasonings. Examples: "
            f"{bad_keys}."
        )

    return out.reset_index(drop=True)


def require_schema_columns(
    df: pd.DataFrame,
    required: List[str],
    context: str,
    min_schema_version: int = SCHEMA_VERSION,
) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: missing schema columns {missing}. "
            "Legacy results are not supported; rerun Phase 1."
        )
    if "schema_version" in df.columns:
        version = int(df["schema_version"].dropna().iloc[0])
        if version < min_schema_version:
            raise ValueError(
                f"{context}: schema_version={version} < required {min_schema_version}. "
                "Rerun earlier phases."
            )


def load_split_manifest(
    path: str = DEFAULT_SPLIT_MANIFEST,
    expected_thesis_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load and validate the published, fixed train/selection assignment."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype={"thesis_nr": str})
    required = [
        "thesis_key",
        "election_id",
        "thesis_nr",
        "major_topic",
        "topic_group",
        "split",
        "thesis",
    ]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise ValueError(f"Split manifest is missing columns: {missing}")
    if manifest["thesis_key"].duplicated().any():
        duplicates = manifest.loc[
            manifest["thesis_key"].duplicated(keep=False), "thesis_key"
        ].unique()
        raise ValueError(f"Split manifest contains duplicate thesis_key values: {duplicates[:5]}")
    invalid_splits = sorted(set(manifest["split"]) - {"train", "selection"})
    if invalid_splits:
        raise ValueError(f"Split manifest contains invalid split labels: {invalid_splits}")
    group_split_counts = manifest.groupby("topic_group")["split"].nunique()
    if (group_split_counts > 1).any():
        bad_groups = group_split_counts[group_split_counts > 1].index.tolist()[:5]
        raise ValueError(
            "Semantically grouped theses cross split boundaries: "
            f"{bad_groups}"
        )
    if expected_thesis_keys is not None:
        expected = set(map(str, expected_thesis_keys))
        actual = set(manifest["thesis_key"].astype(str))
        missing_keys = sorted(expected - actual)
        extra_keys = sorted(actual - expected)
        if missing_keys or extra_keys:
            raise ValueError(
                "Split manifest does not match the workbook. "
                f"Missing={missing_keys[:5]}, extra={extra_keys[:5]}"
            )
    return manifest


def apply_split_manifest(
    df: pd.DataFrame,
    manifest_path: str = DEFAULT_SPLIT_MANIFEST,
    split: Optional[str] = None,
) -> pd.DataFrame:
    """Attach the fixed split and optionally retain one split."""
    if "thesis_key" not in df.columns:
        raise ValueError("Cannot apply split manifest without thesis_key.")
    manifest = load_split_manifest(
        manifest_path,
        expected_thesis_keys=df["thesis_key"].astype(str).unique().tolist(),
    )
    columns = ["thesis_key", "major_topic", "topic_group", "split"]
    out = df.drop(columns=["major_topic", "topic_group", "split"], errors="ignore").merge(
        manifest[columns],
        on="thesis_key",
        how="left",
        validate="many_to_one",
    )
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), "thesis_key"].unique().tolist()[:5]
        raise ValueError(f"Rows are absent from split manifest: {missing}")
    if split is not None:
        if split not in {"train", "selection"}:
            raise ValueError(f"Unknown split '{split}'.")
        out = out[out["split"] == split].copy()
    return out.reset_index(drop=True)


def sample_complete_theses(
    df: pd.DataFrame,
    sample_size: Optional[int],
    seed: int = DEFAULT_THESIS_SPLIT_SEED,
) -> pd.DataFrame:
    """Approximate a row budget while retaining every party row per sampled thesis."""
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)
    if sample_size < 1:
        raise ValueError("SAMPLE_SIZE must be positive.")
    grouped = df.groupby("thesis_key", sort=True).size()
    shuffled_keys = grouped.sample(frac=1.0, random_state=seed).index.tolist()
    selected: List[str] = []
    row_count = 0
    for thesis_key in shuffled_keys:
        group_size = int(grouped.loc[thesis_key])
        if selected and row_count + group_size > sample_size:
            continue
        selected.append(thesis_key)
        row_count += group_size
        if row_count >= sample_size:
            break
    return df[df["thesis_key"].isin(selected)].reset_index(drop=True)


def load_wahlomat_data(path: str, parties: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_excel(path)
    required_cols = ["These: These", "Position: Position", "Position: Begründung", "These: Nr."]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {df.columns.tolist()}")
    df = df[df["Position: Position"].isin(STANCE_MAPPING.keys())].copy()
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        print(
            f"Warning: dropping {duplicate_count} exact duplicate workbook rows; "
            "the source workbook is left unchanged.",
            flush=True,
        )
        df = df.drop_duplicates().copy()
    df["true_stance"] = df["Position: Position"].map(STANCE_MAPPING)
    df["party_reasoning"] = df["Position: Begründung"].astype(str)
    df = add_canonical_keys(df, party_keys=parties)
    print(
        f"Loaded {len(df)} rows from {path} "
        f"({df['thesis_key'].nunique()} unique theses).",
        flush=True,
    )
    return df


def filter_party_df(df: pd.DataFrame, party_name: str) -> pd.DataFrame:
    if "Partei: Kurzbezeichnung" in df.columns:
        return df[df["Partei: Kurzbezeichnung"] == party_name]
    if "Partei: Name" in df.columns:
        return df[df["Partei: Name"] == party_name]
    return df


def build_thesis_party_reasoning_lookup(
    df: pd.DataFrame,
    party_keys: List[str],
) -> Dict[str, Dict[str, str]]:
    require_schema_columns(df, ["thesis_key", "party_key", "party_reasoning"], "lookup")
    lookup: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        thesis_key = str(row["thesis_key"])
        party_key = row.get("party_key")
        reasoning = str(row.get("party_reasoning", "") or "")
        if party_key not in party_keys:
            continue
        lookup.setdefault(thesis_key, {})[party_key] = reasoning
    return lookup


def get_party_reasonings_for_thesis(
    lookup: Dict[str, Dict[str, str]],
    thesis_key: str,
    party_keys: List[str],
) -> Dict[str, str]:
    reasonings = lookup.get(thesis_key, {})
    missing = [pk for pk in party_keys if not reasonings.get(pk)]
    if missing:
        raise ValueError(
            f"Missing party reasonings for thesis_key={thesis_key}: {missing}"
        )
    return {pk: reasonings[pk] for pk in party_keys}


def load_phase1_results(baseline_dir: str, model_short_name: str, party_key: str) -> pd.DataFrame:
    path = Path(baseline_dir) / f"baseline_results_{model_short_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Phase 1 results not found: {path}")
    df = pd.read_csv(path)
    require_schema_columns(
        df,
        ["party_key", "party_statement_key", "thesis_key", "schema_version", "split"],
        "Phase 1 results",
    )
    party_df = df[df["party_key"] == party_key].copy()
    if party_df.empty:
        raise ValueError(f"No Phase 1 records for party_key={party_key}")
    invalid_splits = sorted(set(party_df["split"].dropna()) - {"train"})
    if invalid_splits or party_df["split"].isna().any():
        raise ValueError(
            "Phase 2 vector extraction may use train rows only; found "
            f"{invalid_splits or ['missing split']} for party_key={party_key}."
        )
    return party_df


def similarity_weight(score: Optional[float]) -> float:
    if score is None or pd.isna(score):
        return 0.0
    try:
        score_val = float(score)
    except (TypeError, ValueError):
        return 0.0
    if score_val < 1 or score_val > 10:
        return 0.0
    return score_val / 10.0


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def create_judge_client(cfg: JudgeConfig) -> OpenAI:
    api_key = cfg.api_key or os.getenv("API-KEY") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("No API key found. Export API-KEY or API_KEY.")
    return OpenAI(base_url=cfg.base_url, api_key=api_key)


def _parse_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if 1 <= score <= 10:
        return score
    return None


def _judge_with_retries(
    client: OpenAI,
    cfg: JudgeConfig,
    prompt: str,
    parse_fn,
    max_retries: int = MAX_JUDGE_RETRIES,
) -> Dict[str, Any]:
    last_raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            _pace_judge_call()
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
                max_tokens=cfg.max_new_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()
            last_raw = content
            parsed = parse_fn(content)
            if parsed is not None:
                return parsed
            print(
                f"  [judge] Attempt {attempt}/{max_retries}: unparseable: {content[:120]}",
                flush=True,
            )
        except Exception as exc:
            print(f"  [judge] Attempt {attempt}/{max_retries} error: {exc}", flush=True)

        if attempt < max_retries:
            wait_time = min(JUDGE_BACKOFF_BASE ** (attempt - 1), JUDGE_BACKOFF_MAX)
            print(
                f"  [judge] Backing off {wait_time}s before retry {attempt + 1}/{max_retries}...",
                flush=True,
            )
            time.sleep(wait_time)

    print(f"  [judge] FAILED after {max_retries} retries.", flush=True)
    return {"score": None, "rationale": "Judge API failed after all retries", "raw": last_raw[:200]}


def _parse_similarity_json(content: str) -> Optional[Dict[str, Any]]:
    start = content.find("{")
    end = content.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    obj = json.loads(content[start:end])
    score = _parse_score(obj.get("score"))
    if score is None:
        return None
    return {"score": score, "rationale": obj.get("rationale", ""), "raw": content}


def judge_similarity(
    client: OpenAI,
    cfg: JudgeConfig,
    thesis: str,
    party_name: str,
    party_reasoning: str,
    response: str,
) -> Dict[str, Any]:
    prompt = build_similarity_judge_prompt(thesis, party_reasoning, response, party_name)
    result = _judge_with_retries(client, cfg, prompt, _parse_similarity_json)
    if result.get("score") is None:
        return {"score": None, "rationale": result.get("rationale", ""), "raw": result.get("raw", "")}
    return result


def judge_similarity_batch(
    client: OpenAI,
    cfg: JudgeConfig,
    items: List[Tuple[str, str, str, str]],
    batch_size: int = JUDGE_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    for chunk_start in range(0, len(items), batch_size):
        chunk = items[chunk_start: chunk_start + batch_size]
        prompt = build_similarity_judge_batch_prompt(chunk)

        def _parse_batch(content: str):
            start = content.find("[")
            end = content.rfind("]") + 1
            if start < 0 or end <= start:
                return None
            parsed = json.loads(content[start:end])
            if not isinstance(parsed, list) or len(parsed) != len(chunk):
                raise ValueError(
                    f"Expected {len(chunk)} items, got "
                    f"{len(parsed) if isinstance(parsed, list) else 'non-list'}"
                )
            results = []
            for obj in parsed:
                score = _parse_score(obj.get("score"))
                results.append(
                    {
                        "score": score,
                        "rationale": obj.get("rationale", ""),
                        "raw": content,
                    }
                )
            return results

        try:
            _pace_judge_call()
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
                max_tokens=cfg.max_new_tokens * len(chunk),
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = _parse_batch(content)
            if parsed is None:
                raise ValueError("Batch parse returned None")
            all_results.extend(parsed)
        except Exception as exc:
            print(
                f"  Batch judge failed [{chunk_start}:{chunk_start + len(chunk)}]: {exc}. "
                "Falling back.",
                flush=True,
            )
            for thesis, party_name, party_reasoning, response in chunk:
                all_results.append(
                    judge_similarity(client, cfg, thesis, party_name, party_reasoning, response)
                )

    none_idx = [i for i, r in enumerate(all_results) if r.get("score") is None]
    if none_idx:
        print(f"  Re-judging {len(none_idx)} item(s) ...", flush=True)
        for i in none_idx:
            thesis, party_name, party_reasoning, response = items[i]
            all_results[i] = judge_similarity(
                client, cfg, thesis, party_name, party_reasoning, response
            )
    return all_results


def _parse_cross_party_json(content: str, party_keys: List[str]) -> Optional[Dict[str, Any]]:
    start = content.find("{")
    end = content.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    obj = json.loads(content[start:end])
    scores_obj = obj.get("scores", {})
    if not isinstance(scores_obj, dict):
        return None
    scores: Dict[str, Optional[float]] = {}
    for pk in party_keys:
        scores[pk] = _parse_score(scores_obj.get(pk))
    if all(v is None for v in scores.values()):
        return None
    return {
        "scores": scores,
        "rationale": obj.get("rationale", ""),
        "raw": content,
    }


def flatten_party_judge_results(
    judge_result: Dict[str, Any],
    party_keys: List[str],
) -> Dict[str, Any]:
    scores = judge_result.get("scores", {})
    rationales = judge_result.get("rationales", {})
    out: Dict[str, Any] = {}
    for pk in party_keys:
        out[f"sim_{pk}"] = scores.get(pk)
        out[f"judge_rationale_{pk}"] = rationales.get(pk, "")
    return out


def judge_all_parties_similarity_batch(
    client: OpenAI,
    cfg: JudgeConfig,
    items: List[Tuple[str, str, str]],
    party_keys: List[str],
    reasoning_lookup: Dict[str, Dict[str, str]],
    batch_size: int = JUDGE_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """Judge each response with five isolated single-party prompts."""
    if not items:
        return []

    n_items = len(items)
    scores: List[Dict[str, Optional[float]]] = [
        {pk: None for pk in party_keys} for _ in range(n_items)
    ]
    rationales: List[Dict[str, str]] = [{pk: "" for pk in party_keys} for _ in range(n_items)]

    for pk in party_keys:
        party_name = PARTIES[pk]
        party_items: List[Tuple[str, str, str, str]] = []
        for thesis, response, thesis_key in items:
            reasonings = get_party_reasonings_for_thesis(reasoning_lookup, thesis_key, [pk])
            party_items.append((thesis, party_name, reasonings[pk], response))

        party_results = judge_similarity_batch(
            client,
            cfg,
            party_items,
            batch_size=batch_size,
        )
        if len(party_results) != n_items:
            raise RuntimeError(
                f"Judge alignment error for {pk}: expected {n_items}, got {len(party_results)}"
            )
        for idx, result in enumerate(party_results):
            scores[idx][pk] = result.get("score")
            rationales[idx][pk] = result.get("rationale", "")

    return [{"scores": scores[i], "rationales": rationales[i]} for i in range(n_items)]


def bootstrap_mean_ci(
    values: List[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = DEFAULT_THESIS_SPLIT_SEED,
) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    series = pd.Series(values, dtype=float)
    if len(series) == 1:
        val = float(series.iloc[0])
        return val, val
    rng = pd.Series(range(n_boot)).sample(n=n_boot, replace=True, random_state=seed)
    means = [
        float(series.sample(n=len(series), replace=True, random_state=int(seed + i)).mean())
        for i in rng
    ]
    lower = float(pd.Series(means).quantile(alpha / 2))
    upper = float(pd.Series(means).quantile(1 - alpha / 2))
    return lower, upper


def paired_delta_summary(
    steered_df: pd.DataFrame,
    alpha0_df: pd.DataFrame,
    target_col: str,
    thesis_col: str = "thesis_key",
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    group_cols = group_cols or ["layer", "alpha"]
    alpha0_map = alpha0_df.set_index(thesis_col)[target_col].to_dict()
    rows = []
    for group_vals, grp in steered_df.groupby(group_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        deltas = []
        for _, row in grp.iterrows():
            thesis_key = row[thesis_col]
            base = alpha0_map.get(thesis_key)
            score = row.get(target_col)
            if base is None or score is None or pd.isna(base) or pd.isna(score):
                continue
            deltas.append(float(score) - float(base))
        ci_low, ci_high = bootstrap_mean_ci(deltas)
        record = dict(zip(group_cols, group_vals))
        record.update(
            {
                "mean_delta": float(sum(deltas) / len(deltas)) if deltas else None,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_pairs": len(deltas),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def select_best_steering_config(
    results_df: pd.DataFrame,
    alpha0_df: pd.DataFrame,
    target_col: str,
    steered_party_key: str,
    split: str = "selection",
) -> Dict[str, Any]:
    split_df = results_df[results_df["split"] == split].copy()
    alpha0_split = alpha0_df[alpha0_df["split"] == split].copy()
    summary = paired_delta_summary(split_df, alpha0_split, target_col=target_col)
    if summary.empty or summary["mean_delta"].notna().sum() == 0:
        return {}
    best = summary.loc[summary["mean_delta"].idxmax()]
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": int(best["layer"]),
        "alpha": float(best["alpha"]),
        "selection_mean_delta": float(best["mean_delta"]),
        "selection_ci_low": best.get("ci_low"),
        "selection_ci_high": best.get("ci_high"),
        "selection_n_pairs": int(best["n_pairs"]),
        "selection_split": split,
        "selection_set_is_final_holdout": False,
        "target_party_key": steered_party_key,
    }


def judge_cross_party_similarity(
    client: OpenAI,
    cfg: JudgeConfig,
    thesis: str,
    party_reasonings: Dict[str, str],
    response: str,
    party_keys: List[str],
) -> Dict[str, Any]:
    prompt = build_cross_party_similarity_judge_prompt(
        thesis, party_reasonings, response, party_keys
    )
    result = _judge_with_retries(
        client,
        cfg,
        prompt,
        lambda content: _parse_cross_party_json(content, party_keys),
    )
    if "scores" not in result:
        return {
            "scores": {pk: None for pk in party_keys},
            "rationale": result.get("rationale", ""),
            "raw": result.get("raw", ""),
        }
    return result


def judge_cross_party_similarity_batch(
    client: OpenAI,
    cfg: JudgeConfig,
    items: List[Tuple[str, Dict[str, str], str]],
    party_keys: List[str],
    batch_size: int = JUDGE_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []

    def _parse_batch(content: str):
        start = content.find("[")
        end = content.rfind("]") + 1
        if start < 0 or end <= start:
            return None
        parsed = json.loads(content[start:end])
        if not isinstance(parsed, list) or len(parsed) != len(chunk):
            raise ValueError(
                f"Expected {len(chunk)} items, got "
                f"{len(parsed) if isinstance(parsed, list) else 'non-list'}"
            )
        results = []
        for obj in parsed:
            scores_obj = obj.get("scores", {})
            scores = {pk: _parse_score(scores_obj.get(pk)) for pk in party_keys}
            results.append(
                {
                    "scores": scores,
                    "rationale": obj.get("rationale", ""),
                    "raw": content,
                }
            )
        return results

    for chunk_start in range(0, len(items), batch_size):
        chunk = items[chunk_start: chunk_start + batch_size]
        prompt = build_cross_party_similarity_batch_prompt(chunk, party_keys)
        try:
            _pace_judge_call()
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
                max_tokens=cfg.max_new_tokens * len(chunk),
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = _parse_batch(content)
            if parsed is None:
                raise ValueError("Batch parse returned None")
            all_results.extend(parsed)
        except Exception as exc:
            print(
                f"  Batch cross-party judge failed "
                f"[{chunk_start}:{chunk_start + len(chunk)}]: {exc}. Falling back.",
                flush=True,
            )
            for thesis, party_reasonings, response in chunk:
                all_results.append(
                    judge_cross_party_similarity(
                        client, cfg, thesis, party_reasonings, response, party_keys
                    )
                )

    failed_idx = [
        i
        for i, r in enumerate(all_results)
        if all(r.get("scores", {}).get(pk) is None for pk in party_keys)
    ]
    if failed_idx:
        print(f"  Re-judging {len(failed_idx)} cross-party item(s) ...", flush=True)
        for i in failed_idx:
            thesis, party_reasonings, response = items[i]
            all_results[i] = judge_cross_party_similarity(
                client, cfg, thesis, party_reasonings, response, party_keys
            )
    return all_results


# ---------------------------------------------------------------------------
# Vector IO
# ---------------------------------------------------------------------------

def load_ideology_vectors(
    vector_dir: str,
    model_short_name: str,
    party_key: str,
) -> Dict[int, torch.Tensor]:
    party_dir = Path(vector_dir) / model_short_name / party_key
    if not party_dir.exists():
        raise FileNotFoundError(f"Vector directory not found: {party_dir}")
    with open(party_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("schema_version", 1) < SCHEMA_VERSION:
        raise ValueError(
            f"Vector metadata schema_version={meta.get('schema_version')} is legacy. "
            "Recompute vectors in Phase 2."
        )
    vectors: Dict[int, torch.Tensor] = {}
    for layer_idx in range(meta["num_layers"]):
        vectors[layer_idx] = torch.load(
            party_dir / f"vector_layer_{layer_idx:02d}.pt", weights_only=True
        )
    print(
        f"Loaded {meta['num_layers']} vectors for {meta['party_name']} ({model_short_name}).",
        flush=True,
    )
    return vectors


def save_ideology_vectors(
    vectors: Dict[int, torch.Tensor],
    output_dir: str,
    model_short_name: str,
    party_key: str,
    model_name: str,
    n_statements: int,
    sum_of_weights: float,
    statement_weights: List[Dict[str, Any]],
    source_results_path: Optional[str] = None,
    weighting_diagnostics: Optional[Dict[str, Any]] = None,
) -> Path:
    party_dir = Path(output_dir) / model_short_name / party_key
    party_dir.mkdir(parents=True, exist_ok=True)
    num_layers = len(vectors)
    for layer_idx, vec in vectors.items():
        torch.save(vec, party_dir / f"vector_layer_{layer_idx:02d}.pt")
    norms = {str(layer): float(vec.norm().item()) for layer, vec in vectors.items()}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "vector_method": "prompt_last_token_party_minus_other_parties_mean",
        "steering_scale": "raw_vector_times_alpha",
        "model": model_name,
        "model_short_name": model_short_name,
        "party_key": party_key,
        "party_name": PARTIES[party_key],
        "num_layers": num_layers,
        "hidden_dim": int(vectors[0].shape[0]),
        "n_statements": n_statements,
        "weighting": "similarity/10",
        "sum_of_weights": sum_of_weights,
        "statement_weights": statement_weights,
        "source_results_path": source_results_path,
        "weighting_diagnostics": weighting_diagnostics or {},
        "timestamp": datetime.now().isoformat(),
        "vector_norms_per_layer": norms,
    }
    with open(party_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return party_dir


def party_keys_from_env(default_all: bool = True) -> List[str]:
    raw = env_list_str("PARTY_KEYS", list(PARTIES.keys()) if default_all else [])
    bad = [p for p in raw if p not in PARTIES]
    if bad:
        raise ValueError(f"Unknown party keys: {bad}. Valid: {list(PARTIES)}")
    return raw
