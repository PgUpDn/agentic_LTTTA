# agentic_lttta

Python package for the Agentic LTTTA baseline prototype.

See the repository-level `README.md` for setup, data layout, competition notes,
and run commands.

LLM usage is split into explicit modes: optional Google ADK agents can run
offline to tune `config/policy.yaml`; the default online evaluator uses the
bounded rule controller and does not call an LLM; and the online physics-advisor
demo can call Gemini/ADK to recommend bounded actions. Official online LLM use,
if enabled by the competition, should go through the organizer API gateway
rather than participant-owned keys.

For an online LLM-agent demonstration, run:

```bash
python -m agentic_lttta.scripts.run_online_llm_demo --n-blocks 20
```

The default demo uses `config/policy_llm_demo.yaml` and a mock organizer gateway,
so it exercises the online gateway path without external credentials.

For the ADK physics-advisor demo, run:

```bash
python -m agentic_lttta.scripts.run_online \
  --policy agentic_lttta/config/policy_physics_advisor_demo.yaml \
  --use-physics-advisor \
  --advisor-model gemini-2.5-pro \
  --n-blocks 6
```
