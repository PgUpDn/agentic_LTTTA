# Agentic LTTTA Baseline

Agentic Long-Term Test-Time Adaptation (LTTTA) baseline prototype for the
RealPDE / NeurIPS 2026 competition Track 2.

This repository contains a bounded online controller for long-horizon
autoregressive foil-flow forecasting. The default baseline policy does not call
an LLM online. The optional Google ADK agents are an offline design layer that
can tune the controller policy before evaluation, and the separate
`llm_gateway` demo mode shows how an official online LLM agent would plug in.

## LLM Usage

This baseline uses an **offline-design / online-execution** split.

During offline development, you may optionally run the Google ADK design agents
to search for a better controller policy. That step can use a local
`GOOGLE_API_KEY` or `GEMINI_API_KEY`, and writes the selected knobs to
`agentic_lttta/config/policy.yaml`.

During default online evaluation, this repository does **not** call an LLM. The
streaming evaluator loads `policy.yaml` and uses the bounded rule controller in
`agentic_lttta/controller.py` to choose actions such as `observe`,
`recalibrate`, `update_adapter`, or `skip_update`. This keeps the main baseline
deterministic, reproducible, and free of participant-owned external API keys in
the scored loop.

If the official competition allows online LLM agents, those calls must be wired
through the organizer-provided API gateway with its fixed model versions, token
limits, timeout rules, logging, and wall-clock accounting. Direct participant
API keys should not be used in official LTTTA evaluation.

### Online LLM Agent Demo

To demonstrate the official-online-LLM setting before the real gateway is
available, run the mock organizer-gateway demo:

```bash
python -m agentic_lttta.scripts.run_online_llm_demo --n-blocks 20
```

This compares three variants:

- no adaptation
- the bounded rule controller
- an online LLM-agent controller through `policy_llm_demo.yaml`

The mock gateway is deterministic and local. It exercises the same interface the
official gateway should use: compact state in, one bounded action out, with
latency counted against the rollout budget and decision logs stored per block.

When the organizer gateway is available, switch the demo policy to:

```yaml
mode: llm_gateway
llm_gateway_provider: organizer_http
llm_gateway_url: https://...
```

and set the organizer-issued gateway credential, if required:

```bash
export LTTTA_GATEWAY_TOKEN=...
```

Do not replace this with a participant-owned Gemini/OpenAI/Anthropic key.

## Baseline Status

This code is a useful agentic-policy reference baseline, but it should be
treated as a prototype rather than the final official starting kit:

- It implements a transparent bounded action loop: observe, recalibrate,
  retrieve memory, select expert, update adapter, or skip.
- It includes an approximate Track-2 metric implementation while the final
  competition scoring constants remain organizer-controlled.
- It relies on a local RealPDEBench checkout and downloaded RealPDE data.
- It does not include the official Codabench submission wrapper.
- External participant API keys are not allowed in official LTTTA evaluation.
  If LLM calls are used in the competition, they must go through the organizer
  gateway and its fixed limits. The ADK layer here is for local offline policy
  design only.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional ADK policy-design tools:

```bash
pip install -e ".[adk]"
export GOOGLE_API_KEY=...
```

These credentials are only for local offline policy design. They are not needed
for `run_baseline` or `run_online`.

Do not commit API keys, `.env` files, downloaded datasets, or generated result
files.

## Data And RealPDEBench

The code expects these paths next to the `agentic_lttta/` package:

```text
RealPDEBench/
dataset_RealPDE/
agentic_lttta/
```

Download a small local data slice:

```bash
python -m agentic_lttta.scripts.download_data --n-shards 1
```

The baseline checkpoint in `agentic_lttta/checkpoints/` can be reused for quick
experiments, or retrained locally.

## Run

Compare no-adaptation against the default adaptive policy:

```bash
python -m agentic_lttta.scripts.run_baseline --n-blocks 20 --train-iters 200
```

Evaluate a policy YAML:

```bash
python -m agentic_lttta.scripts.run_online \
  --policy agentic_lttta/config/policy.yaml \
  --n-blocks 60
```

Demonstrate the online LLM-agent pathway with a local mock organizer gateway:

```bash
python -m agentic_lttta.scripts.run_online_llm_demo --n-blocks 20
```

Run offline ADK-assisted policy design:

```bash
python -m agentic_lttta.adk_agents.run_design --n-blocks 20
```

If no valid API key is available, `run_design` falls back to a deterministic
grid search.

## Repository Hygiene

This repository intentionally excludes:

- `__pycache__/`, `.pyc`, and local editor files
- `.env` and other secret-bearing files
- downloaded `dataset_RealPDE/`
- local `RealPDEBench/` checkouts
- generated `agentic_lttta/results/`

Before public release, add the final competition wrapper, official scoring
notes, and an explicit license.
