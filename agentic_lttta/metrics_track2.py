"""Track-2 evaluation metrics (our portable implementation).

The official competition weights/formula are not public yet, so we implement
faithful, documented approximations and keep every weight/scale explicit and
configurable. The set mirrors the announced Track-2 metrics:

* **Relative L2** -- core field accuracy (u, v)
* **TKE error**   -- turbulent kinetic energy (reuses RealPDEBench ``kinetic_energy``)
* **MVPE**        -- mean velocity-profile error in the wake (streamwise u profiles)
* **Time Score**  -- runtime efficiency (budget fraction remaining)
* **SPS**         -- Safe Prediction Score: uncertainty coverage + interval tightness
* **Composite**   -- weighted aggregate (placeholder weights, see :class:`CompositeConfig`)

Lower is better for the error metrics; higher is better for the scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Sequence, Union

import torch

from .paths import ensure_realpdebench_on_path


# --- error metrics ---------------------------------------------------------
def rel_l2(pred: torch.Tensor, target: torch.Tensor, c: int = 2) -> float:
    p, t = pred[..., :c], target[..., :c]
    b = p.shape[0]
    err = torch.norm((p - t).reshape(b, -1), dim=1)
    nrm = torch.norm(t.reshape(b, -1), dim=1).clamp_min(1e-8)
    return float((err / nrm).mean())


def rmse(pred: torch.Tensor, target: torch.Tensor, c: int = 2) -> float:
    p, t = pred[..., :c], target[..., :c]
    return float(torch.sqrt(torch.mean((p - t) ** 2)))


def tke_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    ensure_realpdebench_on_path()
    from realpdebench.utils.metrics import kinetic_energy

    return float((kinetic_energy(pred) - kinetic_energy(target)).abs().mean())


def mvpe(
    pred: torch.Tensor,
    target: torch.Tensor,
    stations: Sequence[float] = (0.40, 0.55, 0.70, 0.85),
    ch: int = 0,
) -> float:
    """Mean velocity-profile error: time-averaged streamwise (u) profiles at
    several downstream x-stations, scale-normalised by the target profile range.
    """
    b, t, h, w, c = pred.shape
    errs = []
    for f in stations:
        x = min(int(f * w), w - 1)
        pp = pred[:, :, :, x, ch].mean(dim=1)   # [b, h]
        tt = target[:, :, :, x, ch].mean(dim=1)  # [b, h]
        rng = (tt.amax(dim=1) - tt.amin(dim=1)).clamp_min(1e-6)
        errs.append((pp - tt).abs().mean(dim=1) / rng)
    return float(torch.stack(errs, dim=1).mean())


# --- scores ----------------------------------------------------------------
def time_score(used_time_s: float, time_budget_s: float) -> float:
    """1.0 if instant, ->0 as runtime approaches/exceeds the budget."""
    if time_budget_s <= 0:
        return 0.0
    return float(max(0.0, 1.0 - used_time_s / time_budget_s))


def safe_prediction_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma: Union[float, torch.Tensor],
    nominal: float = 0.95,
    z: float = 1.96,
    c: int = 2,
) -> Dict[str, float]:
    """Reward calibrated *and* tight predictive intervals.

    Intervals are ``pred ± z*sigma``. ``coverage`` is the fraction of true values
    inside; ``coverage_score`` penalises deviation from the nominal level;
    ``tight_score`` rewards narrow intervals (relative to the per-channel target
    std). SPS = coverage_score * tight_score.
    """
    p, t = pred[..., :c], target[..., :c]
    if isinstance(sigma, (int, float)):
        sig = torch.full((c,), float(sigma), device=p.device)
    else:
        sig = sigma[:c].to(p.device).float()
    half = z * sig.view(*([1] * (p.dim() - 1)), c)
    lo, hi = p - half, p + half
    coverage = float(((t >= lo) & (t <= hi)).float().mean())
    coverage_score = max(0.0, 1.0 - abs(coverage - nominal))
    tstd = t.reshape(-1, c).std(dim=0).clamp_min(1e-6)
    width_norm = float((2.0 * half.reshape(-1, c).mean(dim=0) / tstd).mean())
    tight_score = 1.0 / (1.0 + width_norm)
    return {
        "sps": coverage_score * tight_score,
        "coverage": coverage,
        "coverage_score": coverage_score,
        "width_norm": width_norm,
        "tight_score": tight_score,
    }


# --- composite -------------------------------------------------------------
@dataclass
class CompositeConfig:
    """Weights + error->score scales for the composite.

    NOTE: placeholder values pending the official competition formula. Scores are
    ``exp(-error / scale)`` in [0, 1]; the composite is a weight-normalised sum.
    """

    w_acc: float = 0.35      # relative L2
    w_mvpe: float = 0.20     # mean velocity-profile error
    w_tke: float = 0.15      # turbulent kinetic energy
    w_time: float = 0.15     # runtime efficiency
    w_sps: float = 0.15      # safe prediction score
    scale_rel_l2: float = 0.30
    scale_mvpe: float = 0.50
    scale_tke: float = 0.02


def _err_to_score(err: float, scale: float) -> float:
    return float(math.exp(-max(0.0, err) / max(1e-9, scale)))


def compute_composite(
    metrics: Dict[str, float], cfg: CompositeConfig = CompositeConfig()
) -> Dict[str, float]:
    acc_s = _err_to_score(metrics["rel_l2"], cfg.scale_rel_l2)
    mvpe_s = _err_to_score(metrics["mvpe"], cfg.scale_mvpe)
    tke_s = _err_to_score(metrics["tke"], cfg.scale_tke)
    time_s = float(metrics.get("time_score", 0.0))
    sps_s = float(metrics.get("sps", 0.0))
    w = cfg.w_acc + cfg.w_mvpe + cfg.w_tke + cfg.w_time + cfg.w_sps
    composite = (
        cfg.w_acc * acc_s
        + cfg.w_mvpe * mvpe_s
        + cfg.w_tke * tke_s
        + cfg.w_time * time_s
        + cfg.w_sps * sps_s
    ) / max(1e-9, w)
    return {
        "composite": composite,
        "acc_score": acc_s,
        "mvpe_score": mvpe_s,
        "tke_score": tke_s,
        "time_score": time_s,
        "sps_score": sps_s,
    }
