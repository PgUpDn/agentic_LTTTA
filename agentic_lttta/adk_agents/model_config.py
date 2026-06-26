"""Per-agent Gemini model configuration + model listing for the ADK web apps.

Lets each agent (lead / analyst / tuner / physics advisor) use a *different*
Gemini model, selected in ``config/models.yaml``. For local experiments, provide
credentials through ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``.

This module intentionally does **not** modify the existing single-model
``agents.py``; it provides config-driven builders used by the ADK web apps and a
helper to list the models available to the key.
"""

from __future__ import annotations

import os
from typing import List, Optional

import yaml
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

from ..llm_config import DEFAULT_MODEL, PREFERRED_MODELS, configure_env
from ..paths import CONFIG_DIR
from .agents import (
    ANALYST_INSTRUCTION,
    LEAD_INSTRUCTION,
    PHYSICS_ADVISOR_INSTRUCTION,
    TUNER_INSTRUCTION,
)
from .online_advisor_tools import ONLINE_ADVISOR_TOOLS
from .tools import ANALYST_TOOLS, TUNER_TOOLS

DEFAULTS = {
    "lead_model": DEFAULT_MODEL,
    "analyst_model": DEFAULT_MODEL,
    "tuner_model": DEFAULT_MODEL,
    "physics_advisor_model": "gemini-2.5-flash",
}


def models_yaml_path() -> str:
    return os.path.join(CONFIG_DIR, "models.yaml")


def load_model_config(path: Optional[str] = None) -> dict:
    """Read per-agent model names from ``config/models.yaml`` (with defaults)."""
    cfg = dict(DEFAULTS)
    path = path or models_yaml_path()
    if os.path.exists(path):
        try:
            with open(path) as f:
                loaded = yaml.safe_load(f) or {}
            cfg.update({k: loaded[k] for k in cfg if loaded.get(k)})
        except Exception:  # noqa: BLE001
            pass
    return cfg


def make_analyst(model: str) -> LlmAgent:
    return LlmAgent(
        name="rollout_analyst",
        model=model,
        description="Characterises the LTTTA benchmark, baseline, and useful actions.",
        instruction=ANALYST_INSTRUCTION,
        tools=ANALYST_TOOLS,
    )


def make_tuner(model: str) -> LlmAgent:
    return LlmAgent(
        name="policy_tuner",
        model=model,
        description="Searches controller knobs to maximise composite and saves the best policy.",
        instruction=TUNER_INSTRUCTION,
        tools=TUNER_TOOLS,
    )


def make_physics_advisor(model: str) -> LlmAgent:
    return LlmAgent(
        name="online_physics_advisor",
        model=model,
        description="Compares saved policy rollouts with and without the online physics advisor.",
        instruction=PHYSICS_ADVISOR_INSTRUCTION,
        tools=ONLINE_ADVISOR_TOOLS,
    )


def make_lead(
    model: str,
    analyst: LlmAgent,
    tuner: LlmAgent,
    physics_advisor: LlmAgent,
) -> LlmAgent:
    return LlmAgent(
        name="lttta_design_lead",
        model=model,
        description="Coordinates LTTTA policy design across specialist agents.",
        instruction=LEAD_INSTRUCTION,
        tools=[
            AgentTool(agent=analyst),
            AgentTool(agent=tuner),
            AgentTool(agent=physics_advisor),
        ],
    )


def build_team(cfg: Optional[dict] = None):
    """Build (lead, analyst, tuner, physics_advisor) with per-agent models."""
    configure_env()
    cfg = cfg or load_model_config()
    analyst = make_analyst(cfg["analyst_model"])
    tuner = make_tuner(cfg["tuner_model"])
    physics_advisor = make_physics_advisor(cfg["physics_advisor_model"])
    lead = make_lead(cfg["lead_model"], analyst, tuner, physics_advisor)
    return lead, analyst, tuner, physics_advisor


def list_models(key: Optional[str] = None) -> List[str]:
    """Return the Gemini model ids available to the configured API key."""
    key = configure_env(key)
    try:
        from google import genai

        client = genai.Client(api_key=key)
        names = [str(getattr(m, "name", "")).split("/")[-1] for m in client.models.list()]
        return sorted({n for n in names if n.startswith("gemini")})
    except Exception:  # noqa: BLE001
        return list(PREFERRED_MODELS)
