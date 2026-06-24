"""Run the Google-ADK multi-agent design team and emit a tuned ``policy.yaml``.

Flow:
1. Wire the Gemini key from ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``.
2. ``init_tools`` -> load foil data + train/cache the surrogate.
3. Build the lead/analyst/tuner agents and run them with an ``InMemoryRunner``.
4. If (and only if) the agents fail to save a policy, fall back to a small
   deterministic grid search so the artifact is always produced.
5. Print the tuned policy's metrics vs the no-adaptation baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import time

from ..controller import DEFAULT_POLICY
from ..experiment import ExperimentSettings
from ..llm_config import DEFAULT_MODEL, configure_env, smoke_test
from ..paths import CONFIG_DIR, RESULTS_DIR
from . import tools as T

APP_NAME = "lttta_design"
USER_ID = "designer"
KICKOFF = (
    "Design the best Long-Term Test-Time Adaptation controller policy for the "
    "foil wake benchmark and SAVE it. Follow your 3 steps and finish with a "
    "concise report."
)


def _print_event(ev) -> None:
    author = getattr(ev, "author", "?")
    content = getattr(ev, "content", None)
    if not content:
        return
    for part in getattr(content, "parts", []) or []:
        txt = getattr(part, "text", None)
        if txt and txt.strip():
            print(f"  [{author}] {txt.strip()}")
        fc = getattr(part, "function_call", None)
        if fc is not None:
            try:
                args = dict(fc.args)
            except Exception:  # noqa: BLE001
                args = fc.args
            print(f"  [{author}] -> CALL {fc.name}({args})")
        fr = getattr(part, "function_response", None)
        if fr is not None:
            snippet = json.dumps(fr.response, default=str)[:240]
            print(f"  [{author}] <- {fr.name} => {snippet}")


def _is_transient(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(t in s for t in ("503", "unavailable", "429", "overloaded", "resource_exhausted", "deadline"))


def run_agents(model: str) -> bool:
    """Run the ADK team once. Returns True if a policy was saved by the agents."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from .agents import build_agents

    lead, _analyst, _tuner = build_agents(model=model)
    runner = InMemoryRunner(agent=lead, app_name=APP_NAME)
    session = asyncio.run(
        runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    )
    msg = types.Content(role="user", parts=[types.Part.from_text(text=KICKOFF)])

    marker = os.path.join(RESULTS_DIR, "tuned_policy_result.json")
    t0 = time.time()
    print("\n--- ADK design transcript ---")
    for ev in runner.run(user_id=USER_ID, session_id=session.id, new_message=msg):
        _print_event(ev)
    print(f"--- agents finished in {time.time() - t0:.1f}s ---\n")
    return os.path.exists(marker) and os.path.getmtime(marker) >= t0


def run_agents_with_retries(model: str, max_retries: int = 3) -> bool:
    """Run the agents, retrying transient Gemini errors (503/429/overloaded)."""
    for attempt in range(1, max_retries + 1):
        try:
            return run_agents(model)
        except Exception as exc:  # noqa: BLE001
            transient = _is_transient(exc)
            print(f"[warn] agent attempt {attempt}/{max_retries} failed "
                  f"({type(exc).__name__}): {str(exc)[:160]}")
            if transient and attempt < max_retries:
                back = 5 * attempt
                print(f"[warn] transient error; retrying in {back}s ...")
                time.sleep(back)
                continue
            return False
    return False


def grid_search_and_save() -> dict:
    """Deterministic fallback: small grid search over the key knobs."""
    print("[fallback] running deterministic grid search ...")
    best = None
    grid = itertools.product(
        [0.06, 0.10],          # err_low
        [0.16, 0.24],          # err_high
        [True, False],         # use_observe
        [True, False],         # use_adapter
        [2, 4],                # adapt_steps
        ["bn", "bn+head"],     # adapt_scope
    )
    for el, eh, uo, ua, ks, sc in grid:
        res = T.evaluate_policy_tool(
            err_low=el, err_high=eh, use_observe=uo, use_adapter=ua,
            adapt_steps=ks, adapt_scope=sc,
        )
        comp = res["metrics"]["composite"]
        if best is None or comp > best["composite"]:
            best = {"composite": comp, "params": dict(
                err_low=el, err_high=eh, use_observe=uo, use_adapter=ua,
                adapt_steps=ks, adapt_scope=sc)}
    saved = T.save_policy(note="deterministic grid-search fallback", **best["params"])
    print(f"[fallback] best composite={best['composite']:.4f} saved -> {saved['saved_to']}")
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-blocks", type=int, default=20)
    ap.add_argument("--train-iters", type=int, default=300)
    ap.add_argument("--max-retries", type=int, default=3, help="retries for transient Gemini errors")
    ap.add_argument("--no-agent", action="store_true", help="skip the LLM agents, grid-search only")
    args = ap.parse_args()

    configure_env()
    sm = smoke_test(model=args.model)
    print(f"[llm] smoke_test: {sm}")

    print("[setup] loading data + surrogate (train/cache) ...")
    T.init_tools(ExperimentSettings(n_blocks=args.n_blocks, train_iters=args.train_iters))

    baseline = T.get_setup_info()["no_adaptation_baseline"]
    print(f"[baseline] no-adaptation: {baseline['metrics']}")

    saved_by_agent = False
    if not args.no_agent and sm.get("ok"):
        saved_by_agent = run_agents_with_retries(args.model, max_retries=args.max_retries)
    if not saved_by_agent:
        grid_search_and_save()

    # Report the final saved policy vs baseline.
    import yaml

    with open(os.path.join(CONFIG_DIR, "policy.yaml")) as f:
        tuned = yaml.safe_load(f)
    tuned_metrics = T._run(tuned)["metrics"]
    print("\n=== FINAL ===")
    print(f"baseline composite : {baseline['metrics']['composite']:.4f}")
    print(f"tuned    composite : {tuned_metrics['composite']:.4f}")
    print(f"tuned policy saved : {os.path.join(CONFIG_DIR, 'policy.yaml')}")
    print(json.dumps({"baseline": baseline["metrics"], "tuned": tuned_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
