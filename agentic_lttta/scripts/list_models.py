"""Print the Gemini models available to the configured key + the per-agent config.

    python -m agentic_lttta.scripts.list_models
"""

from __future__ import annotations

import json

from ..adk_agents.model_config import list_models, load_model_config


def main() -> int:
    print("Configured per-agent models (agentic_lttta/config/models.yaml):")
    print(json.dumps(load_model_config(), indent=2))
    print("\nGemini models available to this API key:")
    for m in list_models():
        print("  ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
