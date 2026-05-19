"""TuRBO-1 with q=1 for the BBO capstone (one query per function per week).

Adapted from Eriksson et al. (NeurIPS 2019) and the BoTorch turbo_1 tutorial,
extended with multi-kernel Thompson sampling (Optuna BBO 2020 entry, Shibata
et al.). Instead of fitting one GP with a fixed kernel, we fit multiple GPs
across kernels {Matern 0.5, 1.5, 2.5, RBF} and draw Thompson samples from
all of them at the same shared candidates. The argmax across the (kernel,
candidate) grid selects the next query. The trust region is built from the
canonical Matern 2.5 lengthscales (same TR shape as W7) so only the posterior
mixture changes.

The trust-region state (length L, success/failure counters, best point) persists
across weeks via JSON, so each weekly call resumes the state machine cleanly.

Usage:
    state = load_state(n=5, default_for_d=4)
    state = update_state(state, y_new=...)
    candidate, info = generate_candidate(state, X_all, Y_all)
    save_state(n=5, state=state)
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

DEVICE = torch.device("cpu")
DTYPE = torch.double

# Canonical kernel used for trust-region ARD lengthscales (matches W7 single-kernel).
_CANONICAL_KERNEL = "Matern25"


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
    failure_tolerance: Optional[int] = None
    best_value: float = -float("inf")
    restart_count: int = 0

    def __post_init__(self):
        if self.failure_tolerance is None:
            self.failure_tolerance = int(math.ceil(max(4.0, float(self.dim))))


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _state_path(n: int) -> Path:
    return _PROJECT_ROOT / f"data/function_{n}/turbo_state.json"


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

    A success requires improvement greater than 1e-3 * max(|best|, 1) over the
    previous best (the standard TuRBO threshold).
    """
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
        state.L = state.L_init
        state.success_counter = 0
        state.failure_counter = 0
        state.restart_count += 1

    return state


def _kernel_factories(d: int):
    """Return (name, factory) pairs. Factories build a fresh ScaleKernel each call."""
    return [
        ("Matern05", lambda: ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=d))),
        ("Matern15", lambda: ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=d))),
        ("Matern25", lambda: ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))),
        ("RBF",      lambda: ScaleKernel(RBFKernel(ard_num_dims=d))),
    ]


def _fit_gp_with_kernel(
    Xt: torch.Tensor, yt: torch.Tensor, covar: ScaleKernel, d: int
) -> SingleTaskGP:
    gp = SingleTaskGP(
        Xt, yt,
        covar_module=covar,
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp.eval()


def _fit_gps(X: np.ndarray, Y: np.ndarray) -> list[tuple[str, SingleTaskGP]]:
    """Fit GPs across the kernel family. Skip kernels that fail to fit."""
    Xt = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
    yt = torch.as_tensor(Y, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    d = X.shape[1]
    fitted: list[tuple[str, SingleTaskGP]] = []
    for name, kfn in _kernel_factories(d):
        try:
            gp = _fit_gp_with_kernel(Xt, yt, kfn(), d)
            fitted.append((name, gp))
        except Exception:
            continue
    if not fitted:
        raise RuntimeError("All GP fits failed in TuRBO multi-kernel ensemble")
    return fitted


def _build_trust_region(
    state: TurboState,
    x_center: np.ndarray,
    gps_by_name: dict[str, SingleTaskGP],
) -> tuple[np.ndarray, np.ndarray]:
    """Hyperrectangle around x_center stretched by ARD lengthscales of the canonical kernel."""
    gp = gps_by_name.get(_CANONICAL_KERNEL) or next(iter(gps_by_name.values()))
    base = gp.covar_module.base_kernel  # ScaleKernel.base_kernel is the inner Matern/RBF
    ls = base.lengthscale.detach().cpu().numpy().reshape(-1)
    weights = ls / np.exp(np.mean(np.log(ls)))
    weights = weights / np.mean(weights)
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
    """Generate the next query via TuRBO-1 with multi-kernel Thompson sampling.

    Fits four GPs (Matern 0.5, 1.5, 2.5, RBF), draws one Thompson sample from each
    at the same shared candidate set inside the trust region, and picks the argmax
    across the (kernel, candidate) grid. Returns (candidate, info_dict).
    """
    rng = np.random.default_rng(seed)
    dim = X.shape[1]

    i_best = int(np.argmax(Y))
    x_center = X[i_best].copy()

    fitted = _fit_gps(X, Y)
    gps_by_name = dict(fitted)

    lo, hi = _build_trust_region(state, x_center, gps_by_name)

    cand_np = rng.uniform(lo, hi, size=(n_candidates, dim))
    prob_perturb = min(20.0 / dim, 1.0)
    mask = rng.uniform(size=(n_candidates, dim)) < prob_perturb
    no_perturb = mask.sum(axis=1) == 0
    if no_perturb.any():
        rand_dims = rng.integers(0, dim, size=no_perturb.sum())
        mask[no_perturb, rand_dims] = True
    cand_np = np.where(mask, cand_np, x_center)

    cand = torch.as_tensor(cand_np, dtype=DTYPE, device=DEVICE)

    draws = []
    kernel_names = []
    with torch.no_grad():
        for name, gp in fitted:
            posterior = gp.posterior(cand)
            ts = posterior.rsample(sample_shape=torch.Size([1])).squeeze(0).squeeze(-1)
            draws.append(ts)
            kernel_names.append(name)

    # stacked shape (K, N) — find the global argmax across (kernel, candidate)
    stacked = torch.stack(draws, dim=0)
    flat_argmax = int(stacked.flatten().argmax())
    k_idx = flat_argmax // n_candidates
    c_idx = flat_argmax % n_candidates
    pick = cand_np[c_idx]
    winning_kernel = kernel_names[k_idx]

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
        "kernels_fit": kernel_names,
        "winning_kernel": winning_kernel,
    }
    return pick, info
