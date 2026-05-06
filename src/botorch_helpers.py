"""BoTorch-based candidate generators for the BBO capstone.

Provides:
- `gp_ucb_candidate`: SingleTaskGP fit + UCB acquisition with kappa schedule.
- `gp_ei_candidate`: SingleTaskGP fit + qNoisyExpectedImprovement.

Both use Normalize/Standardize input/outcome transforms so we don't have to
preprocess the raw data. Designed for q=1 sequential queries.

Usage:
    cand, pred_mean, pred_std = gp_ucb_candidate(X, Y, week=7, total_weeks=12)
    cand, pred_ei = gp_ei_candidate(X, Y)
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
from botorch.acquisition import (
    UpperConfidenceBound,
    qLogNoisyExpectedImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

DEVICE = torch.device("cpu")
DTYPE = torch.double


def _fit_gp(X: np.ndarray, Y: np.ndarray) -> SingleTaskGP:
    Xt = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
    Yt = torch.as_tensor(Y, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    d = X.shape[1]
    gp = SingleTaskGP(
        Xt, Yt,
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(mll)
    gp.eval()
    return gp


def kappa_schedule(week: int, total_weeks: int = 12, beta_start: float = 2.0, beta_end: float = 0.5) -> float:
    """Exponential decay from explorative to exploitative.

    week=1 → beta_start (more explore), week=total_weeks → beta_end (exploit).
    Formula: beta = beta_start * (beta_end / beta_start) ** ((week-1)/(total_weeks-1)).
    """
    if total_weeks <= 1:
        return beta_end
    ratio = beta_end / beta_start
    t = (week - 1) / (total_weeks - 1)
    return beta_start * (ratio ** t)


def gp_ucb_candidate(
    X: np.ndarray,
    Y: np.ndarray,
    week: int,
    total_weeks: int = 12,
    num_restarts: int = 10,
    raw_samples: int = 512,
) -> tuple[np.ndarray, float, dict]:
    """Generate next query via SingleTaskGP + UCB with kappa schedule.

    Returns (candidate, predicted_y_at_candidate, info).
    """
    gp = _fit_gp(X, Y)
    beta = kappa_schedule(week, total_weeks)
    acq = UpperConfidenceBound(gp, beta=beta)
    bounds = torch.stack([
        torch.zeros(X.shape[1], dtype=DTYPE, device=DEVICE),
        torch.ones(X.shape[1], dtype=DTYPE, device=DEVICE),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cand_t, _ = optimize_acqf(
            acq, bounds=bounds, q=1,
            num_restarts=num_restarts, raw_samples=raw_samples,
        )
    cand = cand_t.detach().cpu().numpy().reshape(-1)

    # Posterior mean at candidate (for diagnostics)
    with torch.no_grad():
        posterior = gp.posterior(cand_t)
        pred_mean = float(posterior.mean.squeeze().item())
        pred_std = float(posterior.variance.sqrt().squeeze().item())

    info = {"beta": beta, "pred_mean": pred_mean, "pred_std": pred_std,
            "ucb_value": pred_mean + np.sqrt(beta) * pred_std,
            "x_best_observed": X[int(np.argmax(Y))].tolist()}
    return cand, pred_mean, info


def gp_ei_candidate(
    X: np.ndarray,
    Y: np.ndarray,
    num_restarts: int = 10,
    raw_samples: int = 512,
) -> tuple[np.ndarray, float, dict]:
    """Generate next query via SingleTaskGP + qLogNoisyExpectedImprovement.

    qLogNEI is the modern default for noisy/uncertain BO settings (HEBO etc.).
    """
    gp = _fit_gp(X, Y)
    Xt = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
    acq = qLogNoisyExpectedImprovement(gp, X_baseline=Xt, prune_baseline=True)
    bounds = torch.stack([
        torch.zeros(X.shape[1], dtype=DTYPE, device=DEVICE),
        torch.ones(X.shape[1], dtype=DTYPE, device=DEVICE),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cand_t, ei_value = optimize_acqf(
            acq, bounds=bounds, q=1,
            num_restarts=num_restarts, raw_samples=raw_samples,
        )
    cand = cand_t.detach().cpu().numpy().reshape(-1)
    with torch.no_grad():
        posterior = gp.posterior(cand_t)
        pred_mean = float(posterior.mean.squeeze().item())

    info = {"ei_value": float(ei_value.item()), "pred_mean": pred_mean,
            "x_best_observed": X[int(np.argmax(Y))].tolist()}
    return cand, pred_mean, info
