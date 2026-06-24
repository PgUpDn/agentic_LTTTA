"""Agentic Long-Term Test-Time Adaptation (LTTTA) for the RealPDE competition.

A portable prototype of a *bounded-action* test-time-adaptation controller for
long-horizon autoregressive PDE forecasting, plus an offline **Google ADK**
multi-agent "design team" that composes and tunes the online controller policy.

The package is intentionally framework-light: it reuses a couple of modules from
the local RealPDEBench checkout (the ``FNO3d`` surrogate and the evaluation
metrics) via ``sys.path`` rather than pip-installing the whole benchmark.

See ``agentic_lttta/docs/`` for the problem statement and per-step manuals.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
