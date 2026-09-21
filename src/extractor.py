"""
Extraction functions: call the LLM API (Anthropic OR OpenAI) for Model A and Model B,
parse the returned JSON, and apply ontology validation for Model B.

Supports two providers via environment variable PROVIDER, or via set_provider():
    PROVIDER=anthropic   uses Claude (default)
    PROVIDER=openai      uses GPT
"""

import os, json, time, re
from ontology import filter_triples
from prompts import (
    MODEL_A_SYSTEM, model_a_user_prompt,
    MODEL_B_SYSTEM, model_b_user_prompt,
)

# ── Configuration ────────────────────────────────────────────────────────────
# Default model per provider; can be overridden by set_model()
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai":    "gpt-4o-mini",
}

# Active configuration (mutable via set_provider/set_model)
_CONFIG = {
    "provider": os.environ.get("PROVIDER", "anthropic").lower(),
    "model":    None,   # resolved lazily — uses DEFAULT_MODELS if not overridden
}

MAX_TOK     = 512
MAX_RETRIES = 3
RETRY_DELAY = 2

_client = None
_anthropic = None
_openai    = None


def set_provider(provider: str):
    """Switch between 'anthropic' and 'openai'. Resets the client."""
    global _client
    provider = provider.lower().strip()
    if provider not in ("anthropic", "openai"):
        raise ValueError(f"Unknown provider: {provider!r}. Use 'anthropic' or 'openai'.")
    _CONFIG["provider"] = provider
    _client = None  # Force re-initialisation on next call


def set_model(model_name: str):
    """Override the default model for the current provider."""
    _CONFIG["model"] = model_name


def get_provider() -> str:
    return _CONFIG["provider"]


def get_model() -> str:
    return _CONFIG["model"] or DEFAULT_MODELS[_CONFIG["provider"]]


# ── Client initialisation ────────────────────────────────────────────────────
def get_client():
    """Initialise the API client for the active provider. Cached."""
    global _client, _anthropic, _openai
    if _client is not None:
        return _client

    provider = _CONFIG["provider"]

    if provider == "anthropic":
        if _anthropic is None:
            import anthropic
            _anthropic = anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set.\n"
                "  Windows:  set ANTHROPIC_API_KEY=sk-ant-...\n"
                "  Mac/Linux: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        _client = _anthropic.Anthropic(api_key=api_key)

    elif provider == "openai":
        if _openai is None:
            import openai
            _openai = openai
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set.\n"
                "  Windows:  set OPENAI_API_KEY=sk-...\n"
                "  Mac/Linux: export OPENAI_API_KEY=sk-..."
            )
        _client = _openai.OpenAI(api_key=api_key)

    return _client


# ── API call ─────────────────────────────────────────────────────────────────
def _call_api(system: str, user: str) -> str:
    """Raw API call with retry on transient errors. Returns raw text."""
    client   = get_client()
    provider = _CONFIG["provider"]
    model    = get_model()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider == "anthropic":
                msg = client.messages.create(
                    model=model,
                    max_tokens=MAX_TOK,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return msg.content[0].text

            else:  # openai
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=MAX_TOK,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                )
                return resp.choices[0].message.content

        except Exception as e:
            # Retry on transient errors (rate limits, 5xx); raise on permanent failures
            name = type(e).__name__
            is_transient = (
                "RateLimit"     in name or
                "ServiceUnavailable" in name or
                "InternalServerError" in name or
                "APIConnectionError" in name or
                "Timeout" in name or
                "APIStatusError" in name and getattr(e, "status_code", 0) >= 500
            )
            if attempt < MAX_RETRIES and is_transient:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


# ── JSON parsing ─────────────────────────────────────────────────────────────
def _parse_json(raw: str) -> list:
    """
    Extract a JSON array from the model's response.
    Handles: plain JSON, markdown code fences, leading/trailing noise.
    """
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end == -1:
        return []

    try:
        triples = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []

    normalised = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        norm = {
            "head":      str(t.get("head",      t.get("head_entity", t.get("subject",  "")))).strip(),
            "head_type": str(t.get("head_type", t.get("subject_type", ""))).strip(),
            "relation":  str(t.get("relation",  t.get("relationship", t.get("predicate", "")))).strip(),
            "tail":      str(t.get("tail",      t.get("tail_entity", t.get("object",   "")))).strip(),
            "tail_type": str(t.get("tail_type", t.get("object_type", ""))).strip(),
        }
        if norm["head"] and norm["tail"] and norm["relation"]:
            normalised.append(norm)
    return normalised


# ── Model A / Model B ────────────────────────────────────────────────────────
def extract_model_a(sentence: str) -> dict:
    """Run Model A (standalone LLM) on a single sentence."""
    raw     = _call_api(MODEL_A_SYSTEM, model_a_user_prompt(sentence))
    triples = _parse_json(raw)
    return {
        "sentence": sentence,
        "model":    f"A_standalone_{_CONFIG['provider']}",
        "triples":  triples,
        "raw":      raw,
    }


def extract_model_b(sentence: str) -> dict:
    """Run Model B (ontology-guided LLM) on a single sentence."""
    raw        = _call_api(MODEL_B_SYSTEM, model_b_user_prompt(sentence))
    raw_trips  = _parse_json(raw)
    valid, rej = filter_triples(raw_trips)
    return {
        "sentence":         sentence,
        "model":            f"B_ontology_guided_{_CONFIG['provider']}",
        "triples":          valid,
        "rejected_triples": rej,
        "raw":              raw,
    }
