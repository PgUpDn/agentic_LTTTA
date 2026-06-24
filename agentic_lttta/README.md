# agentic_lttta

Python package for the Agentic LTTTA baseline prototype.

See the repository-level `README.md` for setup, data layout, competition notes,
and run commands.

LLM usage is split into two stages: the optional Google ADK agents can run
offline to tune `config/policy.yaml`, while the online evaluator uses only the
bounded rule controller and does not call an LLM or require an API key. Official
online LLM use, if enabled by the competition, should go through the organizer
API gateway rather than participant-owned keys.
