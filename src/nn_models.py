"""Neural network surrogates for BBO capstone (Week 04).

One architecture family: 2-hidden-layer MLP with Tanh activation.
Variants differ only by regularization (dropout / weight decay / ensemble).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

Variant = Literal["plain", "dropout", "wd", "ensemble"]
VARIANTS: tuple[Variant, ...] = ("plain", "dropout", "wd", "ensemble")
DEVICE = torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 32, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_in, hidden), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_one(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray | None,
    y_va: np.ndarray | None,
    hidden: int,
    dropout: float,
    weight_decay: float,
    seed: int,
    max_epochs: int = 1000,
    patience: int = 100,
    lr: float = 1e-2,
) -> tuple[MLP, float]:
    """Train a single MLP. Returns (model, best_val_rmse) on scaled y.

    If X_va is None, uses training loss for early stopping (no holdout).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(X_tr.shape[1], hidden=hidden, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    if X_va is not None:
        Xv = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
        yv = torch.tensor(y_va, dtype=torch.float32, device=DEVICE)

    best_rmse = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience_ctr = 0
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            if X_va is not None:
                rmse = torch.sqrt(loss_fn(model(Xv), yv)).item()
            else:
                rmse = torch.sqrt(loss_fn(model(Xt), yt)).item()
        if rmse < best_rmse - 1e-6:
            best_rmse = rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_rmse


def _predict_scaled(models: list[MLP], X: np.ndarray) -> np.ndarray:
    """Average predictions from one or more models (scaled space)."""
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(Xt).cpu().numpy())
    return np.mean(preds, axis=0)


def _fit_variant(
    X: np.ndarray,
    y_scaled: np.ndarray,
    variant: Variant,
    hidden: int,
    X_va: np.ndarray | None = None,
    y_va_scaled: np.ndarray | None = None,
    n_seeds: int = 5,
) -> list[MLP]:
    """Fit all models required for a variant. Returns list (length 1, except ensemble)."""
    dropout = 0.2 if variant == "dropout" else 0.0
    wd = 1e-2 if variant in ("wd", "ensemble") else 0.0
    seeds = range(n_seeds) if variant == "ensemble" else [0]
    models = []
    for s in seeds:
        m, _ = _train_one(X, y_scaled, X_va, y_va_scaled, hidden, dropout, wd, s)
        models.append(m)
    return models


def cv_rmse(
    X: np.ndarray,
    y: np.ndarray,
    variant: Variant,
    hidden: int,
    n_splits: int = 5,
    seed: int = 0,
    ensemble_cv_seeds: int = 3,
) -> float:
    """K-fold CV RMSE on original-y scale."""
    n = len(X)
    n_splits = min(n_splits, n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sq_errors = []
    for tr, va in kf.split(X):
        y_mean = y[tr].mean()
        y_std = y[tr].std() + 1e-8
        y_tr_s = (y[tr] - y_mean) / y_std
        n_seeds = ensemble_cv_seeds if variant == "ensemble" else 1
        models = _fit_variant(X[tr], y_tr_s, variant, hidden, n_seeds=n_seeds)
        pred_s = _predict_scaled(models, X[va])
        pred = pred_s * y_std + y_mean
        sq_errors.append((pred - y[va]) ** 2)
    return float(np.sqrt(np.concatenate(sq_errors).mean()))


def fit_final(
    X: np.ndarray,
    y: np.ndarray,
    variant: Variant,
    hidden: int,
) -> tuple[list[MLP], dict]:
    """Train final model(s) on all data. Returns (models, scaling_meta)."""
    y_mean = float(y.mean())
    y_std = float(y.std()) + 1e-8
    y_scaled = (y - y_mean) / y_std
    models = _fit_variant(X, y_scaled, variant, hidden, n_seeds=5)
    meta = {"y_mean": y_mean, "y_std": y_std, "variant": variant, "hidden": hidden, "d_in": int(X.shape[1])}
    return models, meta


def predict(models: list[MLP], meta: dict, X: np.ndarray) -> np.ndarray:
    """Predict in original Y scale."""
    pred_s = _predict_scaled(models, X)
    return pred_s * meta["y_std"] + meta["y_mean"]


def gradient_at(models: list[MLP], meta: dict, x_point: np.ndarray) -> np.ndarray:
    """dY/dx at x_point in original-Y scale, averaged across ensemble members."""
    x_point = np.asarray(x_point, dtype=np.float32).reshape(1, -1)
    grads = []
    for m in models:
        m.eval()
        xt = torch.tensor(x_point, requires_grad=True, device=DEVICE)
        out = m(xt).sum()
        out.backward()
        grads.append(xt.grad.detach().cpu().numpy().flatten())
    return np.mean(grads, axis=0) * meta["y_std"]


def save(models: list[MLP], meta: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dicts = [m.state_dict() for m in models]
    torch.save({"state_dicts": state_dicts, "meta": meta}, path)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump({k: v for k, v in meta.items() if k not in ("y_mean", "y_std")}
                  | {"y_mean": meta["y_mean"], "y_std": meta["y_std"],
                     "n_ensemble": len(state_dicts)}, f, indent=2)


def load_nn(n: int, models_dir: str | Path = "../models/week_04") -> tuple[list[MLP], dict] | None:
    """Load saved NN for function n. Returns None if not found."""
    path = Path(models_dir) / f"function_{n}_nn.pt"
    if not path.exists():
        return None
    blob = torch.load(path, map_location=DEVICE, weights_only=False)
    meta = blob["meta"]
    dropout = 0.2 if meta["variant"] == "dropout" else 0.0
    models = []
    for sd in blob["state_dicts"]:
        m = MLP(meta["d_in"], hidden=meta["hidden"], dropout=dropout)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models, meta


class NNRegressor:
    """sklearn-compatible wrapper so the NN slots into /analyze Step 4."""

    def __init__(self, models: list[MLP], meta: dict):
        self.models = models
        self.meta = meta

    def predict(self, X: np.ndarray) -> np.ndarray:
        return predict(self.models, self.meta, X)


# --- F1 sign classifier -------------------------------------------------


class MLPClassifier(nn.Module):
    def __init__(self, d_in: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_sign_classifier(
    X: np.ndarray,
    y: np.ndarray,
    hidden: int = 32,
    max_epochs: int = 1000,
    lr: float = 1e-2,
    seed: int = 0,
) -> tuple[MLPClassifier, float]:
    """Train sign(y > 0) classifier. Returns (model, LOO accuracy)."""
    torch.manual_seed(seed)
    labels = (y > 0).astype(np.float32)
    # LOO accuracy
    correct = 0
    for i in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False
        torch.manual_seed(seed)
        m = MLPClassifier(X.shape[1], hidden).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()
        Xt = torch.tensor(X[mask], dtype=torch.float32)
        yt = torch.tensor(labels[mask], dtype=torch.float32)
        for _ in range(max_epochs):
            m.train()
            opt.zero_grad()
            loss = loss_fn(m(Xt), yt)
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(m(torch.tensor(X[i:i+1], dtype=torch.float32))) > 0.5).float().item()
        if pred == labels[i]:
            correct += 1
    loo_acc = correct / len(X)
    # Refit on all data
    torch.manual_seed(seed)
    m = MLPClassifier(X.shape[1], hidden).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(labels, dtype=torch.float32)
    for _ in range(max_epochs):
        m.train()
        opt.zero_grad()
        loss = loss_fn(m(Xt), yt)
        loss.backward()
        opt.step()
    return m, loo_acc


def save_classifier(model: MLPClassifier, loo_acc: float, d_in: int, hidden: int, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "d_in": d_in, "hidden": hidden, "loo_acc": loo_acc}, path)


def load_classifier(n: int, models_dir: str | Path = "../models/week_04") -> tuple[MLPClassifier, float] | None:
    path = Path(models_dir) / f"function_{n}_classifier.pt"
    if not path.exists():
        return None
    blob = torch.load(path, map_location=DEVICE, weights_only=False)
    m = MLPClassifier(blob["d_in"], blob["hidden"])
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m, blob["loo_acc"]
