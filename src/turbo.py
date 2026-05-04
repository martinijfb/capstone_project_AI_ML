"""TuRBO-1 with q=1 for the BBO capstone (one query per function per week).

Adapted from Eriksson et al. (NeurIPS 2019) and the BoTorch turbo_1 tutorial.
The trust-region state (length L, success/failure counters, best point) persists
across weeks via JSON, so each weekly call resumes the state machine cleanly.

Usage:
    state = load_state(n=5, default_for_d=4)
    state = update_state(state, X_new, y_new, X_prev_best_y=2308.0)
    candidate = generate_candidate(state, X_all, Y_all, dim=4)
    save_state(n=5, state=state)
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

DEVICE = torch.device("cpu")
DTYPE = torch.double


@dataclass
class TurboState:
    """Persistent state for TuRBO-1 between weekly queries."""
    dim: int
    L: float = 0.8
    L_init: float = 0.8
    L_min: float = 0.5 ** 7
    L_max: float = 1.6
    success_counter: int = 0
    failure_counter: int = 0
    success_tolerance: int = 3
    failure_tolerance: Optional[int] = None  # filled in __post_init__
    best_value: float = -float("inf")
    restart_count: int = 0

    def __post_init__(self):
        if self.failure_tolerance is None:
            # q=1 case: ceil(max(4 / 1, dim / 1)) = max(4, dim)
            self.failure_tolerance = int(math.ceil(max(4.0, float(self.dim))))


def _state_path(n: int) -> Path:
    return Path(f"data/function_{n}/turbo_state.json")


def load_state(n: int, default_for_d: int) -> TurboState:
    p = _state_path(n)
    if not p.exists():
        return TurboState(dim=default_for_d)
    with p.open() as f:
        return TurboState(**json.load(f))


def save_state(n: int, state: TurboState) -> None:
    p = _state_path(n)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(asdict(state), f, indent=2)


def update_state(state: TurboState, y_new: float) -> TurboState:
    """Apply contract/expand logic given the new observation.

    A success requires improvement greater than 1e-3 * |best| over the
    previous best (the standard TuRBO threshold).
    """
    # Improvement threshold
    threshold = 1e-3 * max(abs(state.best_value), 1.0)
    if y_new > state.best_value + threshold:
        state.success_counter += 1
        state.failure_counter = 0
        state.best_value = float(y_new)
    else:
        state.failure_counter += 1
        state.success_counter = 0

    if state.success_counter >= state.success_tolerance:
        state.L = min(2.0 * state.L, state.L_max)
        state.success_counter = 0
    if state.failure_counter >= state.failure_tolerance:
        state.L = state.L / 2.0
        state.failure_counter = 0

    if state.L < state.L_min:
        # restart triggered: full L back to initial, drop both counters
        state.L = state.L_init
        state.success_counter = 0
        state.failure_counter = 0
        state.restart_count += 1

    return state


def _fit_gp(X: np.ndarray, y: np.ndarray) -> SingleTaskGP:
    Xt = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
    yt = torch.as_tensor(y, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    d = X.shape[1]
    gp = SingleTaskGP(
        Xt, yt,
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp


def _build_trust_region(state: TurboState, x_center: np.ndarray, gp: SingleTaskGP) -> tuple[np.ndarray, np.ndarray]:
    """Hyperrectangle around x_center stretched by ARD lengthscales."""
    ls = gp.covar_module.lengthscale.detach().cpu().numpy().reshape(-1)
    weights = ls / np.exp(np.mean(np.log(ls)))  # geometric mean normalised
    weights = weights / np.mean(weights)  # mean 1.0
    half = state.L * weights / 2.0
    lo = np.clip(x_center - half, 0.0, 1.0)
    hi = np.clip(x_center + half, 0.0, 1.0)
    return lo, hi


def generate_candidate(
    state: TurboState,
    X: np.ndarray,
    Y: np.ndarray,
    n_candidates: int = 5000,
    seed: Optional[int] = 0,
) -> tuple[np.ndarray, dict]:
    """Generate the next query via TuRBO-1 + Thompson sampling (q=1).

    Returns (candidate, info_dict). info_dict contains TR bounds and
    diagnostics for logging.
    """
    rng = np.random.default_rng(seed)
    dim = X.shape[1]

    # Center on argmax of observed Y
    i_best = int(np.argmax(Y))
    x_center = X[i_best].copy()

    # Fit a local GP on data inside the TR (all data if TR is wide)
    # For safety with small data, always include all points.
    gp = _fit_gp(X, Y)
    gp.eval()

    lo, hi = _build_trust_region(state, x_center, gp)

    # Sobol-flavoured candidates via random uniform in TR
    cand_np = rng.uniform(lo, hi, size=(n_candidates, dim))

    # Perturbation mask: sometimes a coordinate stays at center (TuRBO trick)
    prob_perturb = min(20.0 / dim, 1.0)
    mask = rng.uniform(size=(n_candidates, dim)) < prob_perturb
    # Ensure at least one perturbation per row
    no_perturb = mask.sum(axis=1) == 0
    if no_perturb.any():
        rand_dims = rng.integers(0, dim, size=no_perturb.sum())
        mask[no_perturb, rand_dims] = True
    cand_np = np.where(mask, cand_np, x_center)

    cand = torch.as_tensor(cand_np, dtype=DTYPE, device=DEVICE)

    # Thompson sample: draw one realisation from the posterior, pick the max
    with torch.no_grad():
        posterior = gp.posterior(cand)
        ts_draw = posterior.rsample(sample_shape=torch.Size([1])).squeeze(0).squeeze(-1)
    i_pick = int(ts_draw.argmax())
    pick = cand_np[i_pick]

    info = {
        "L": state.L,
        "success_counter": state.success_counter,
        "failure_counter": state.failure_counter,
        "x_center": x_center.tolist(),
        "tr_lo": lo.tolist(),
        "tr_hi": hi.tolist(),
        "n_candidates": n_candidates,
        "restart_count": state.restart_count,
        "best_value": state.best_value,
    }
    return pick, info
