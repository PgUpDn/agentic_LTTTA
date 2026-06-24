"""Gemini / Google ADK credential wiring for optional offline policy design.

No API key is bundled with this repository. For local experiments with the ADK
design agents, provide a key at runtime through ``GOOGLE_API_KEY`` or
``GEMINI_API_KEY``. The online LTTTA controller does not call an LLM.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

# Ordered by preference; the first one that answers a ping wins.
PREFERRED_MODELS: tuple[str, ...] = (
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
)
DEFAULT_MODEL = "gemini-3.5-flash"


def resolve_api_key() -> Optional[str]:
    """Return the Gemini API key from supported environment variables."""
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    return None


def configure_env(key: Optional[str] = None) -> Optional[str]:
    """Export the env vars google-genai / ADK expect (AI-Studio / Gemini API)."""
    key = key or resolve_api_key()
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        # Use the Gemini Developer API (AI Studio), not Vertex AI.
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
    return key


def pick_model(
    preferred: Sequence[str] = PREFERRED_MODELS,
    key: Optional[str] = None,
) -> str:
    """Return the first preferred model that successfully answers a ping."""
    key = configure_env(key)
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY to probe Gemini models.")
    from google import genai

    client = genai.Client(api_key=key)
    last_err: Optional[Exception] = None
    for model in preferred:
        try:
            resp = client.models.generate_content(model=model, contents="ping")
            if resp is not None:
                return model
        except Exception as exc:  # noqa: BLE001 - probe each candidate
            last_err = exc
            continue
    raise RuntimeError(f"No preferred Gemini model is reachable: {last_err}")


def smoke_test(model: Optional[str] = None, key: Optional[str] = None) -> dict:
    """Verify the key works; returns a small diagnostic dict."""
    key = configure_env(key)
    if not key:
        return {"ok": False, "error": "no API key resolved"}
    from google import genai

    client = genai.Client(api_key=key)
    model = model or DEFAULT_MODEL
    try:
        resp = client.models.generate_content(
            model=model, contents="reply with exactly: lttta_ok"
        )
        text = (resp.text or "").strip()
        return {"ok": True, "model": model, "text": text, "key_prefix": key[:6]}
    except Exception as exc:  # noqa: BLE001
        # Fall back to probing the preferred list.
        try:
            model = pick_model(key=key)
            resp = client.models.generate_content(
                model=model, contents="reply with exactly: lttta_ok"
            )
            return {
                "ok": True,
                "model": model,
                "text": (resp.text or "").strip(),
                "key_prefix": key[:6],
            }
        except Exception as exc2:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc2).__name__}: {exc2}"}


if __name__ == "__main__":
    import json

    print(json.dumps(smoke_test(), indent=2))
