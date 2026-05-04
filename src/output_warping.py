"""Yeo-Johnson output warping for skewed-Y functions (F1, F3 etc.).

HEBO (NeurIPS 2020 BBO Challenge winner) showed that Y warping was the
single largest documented gain. We apply PowerTransformer (Yeo-Johnson)
to fit a model on a more Gaussian-shaped output, then invert at predict
time.

Usage:
    wrapper = WarpedRegressor(GradientBoostingRegressor(...))
    wrapper.fit(X, Y)
    y_pred = wrapper.predict(Xnew)            # back in original Y space
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import PowerTransformer


class WarpedRegressor(BaseEstimator, RegressorMixin):
    """Wrap any sklearn regressor with Yeo-Johnson Y warping.

    Fits PowerTransformer on Y (Yeo-Johnson handles negatives), trains the
    wrapped model on warped Y, and inverts the warp on predict.

    Falls back gracefully on tiny / degenerate Y distributions (e.g. all
    near-zero) where Yeo-Johnson fitting fails: in that case it stores no
    transformer and behaves like the base estimator.
    """

    def __init__(self, base_estimator: Any, standardize: bool = True):
        self.base_estimator = base_estimator
        self.standardize = standardize

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = np.asarray(y).reshape(-1)
        self.estimator_ = clone(self.base_estimator)
        try:
            self.transformer_ = PowerTransformer(
                method="yeo-johnson", standardize=self.standardize
            )
            y_warped = self.transformer_.fit_transform(y.reshape(-1, 1)).ravel()
            if not np.isfinite(y_warped).all():
                raise ValueError("non-finite warped Y")
            self.estimator_.fit(X, y_warped)
            self.warped_ = True
        except Exception:
            # fall back to no warping
            self.transformer_ = None
            self.estimator_.fit(X, y)
            self.warped_ = False
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.estimator_.predict(X)
        if self.warped_ and self.transformer_ is not None:
            y_pred = self.transformer_.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return y_pred

    def get_params(self, deep: bool = True) -> dict:
        return {
            "base_estimator": self.base_estimator,
            "standardize": self.standardize,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
