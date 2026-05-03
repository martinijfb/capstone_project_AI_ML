# Black-Box Optimisation Capstone Project

## Section 1: Project Overview

This project tackles the optimisation of 8 unknown black-box functions. The goal is to find the input values that maximise each function's output, with only one query allowed per function per week.

This mirrors real-world scenarios in drug discovery, hyperparameter tuning, and engineering design, where evaluations are expensive and the underlying system is opaque. The challenge is to build surrogate models that approximate the unknown functions well enough to guide the search, while managing the exploration-exploitation trade-off with an extremely limited query budget.

As a professional working with data-driven decision making, this project develops skills directly applicable to my career: building models under uncertainty, validating them rigorously before trusting their predictions, and adapting strategies based on evidence rather than assumptions.

## Section 2: Inputs and Outputs

Each function maps an n-dimensional input vector to a scalar output. The functions vary in dimensionality:

| Function | Dimensions | Initial Points |
|----------|-----------|----------------|
| F1, F2   | 2D        | 10             |
| F3       | 3D        | 15             |
| F4, F5   | 4D        | 20–30          |
| F6       | 5D        | 20             |
| F7       | 6D        | 30             |
| F8       | 8D        | 40             |

**Input format:** `x1-x2-...-xn` where each value is in [0, 1) with exactly 6 decimal places (e.g., `0.700201-0.695377` for a 2D function).

**Output:** A single scalar value. The output range is unspecified and varies significantly across functions (e.g., F1 outputs are near zero at ~1e-16, while F5 outputs exceed 1200).

## Section 3: Challenge Objectives

The goal is to **maximise** each function. Key constraints:

- **1 query per function per week** — 12 total queries over the project
- **Unknown function structure** — no gradient information, no functional form, no output range
- **Increasing dimensionality** — from 2D (tractable for visualisation) to 8D (curse of dimensionality)

The limited budget means brute-force grid search is infeasible. Each query must balance information gain (exploration) against refining known good regions (exploitation).

## Section 4: Technical Approach

My approach centres on a **validate-then-trust** framework: fit multiple surrogate models, validate each via Leave-One-Out Cross-Validation (LOOCV RMSE), and only trust models that beat the baseline (predicting the mean).

**ML methods used:**
- **GridSearchCV with LOOCV** across 7 families: Ridge, KNN, Random Forest, SVR (RBF), Gradient Boosting, Gaussian Process with Matern at ν ∈ {0.5, 1.5, 2.5} and RBF kernels, plus PyTorch MLP surrogates. Kernel smoothness is a CV-chosen hyperparameter, not hard-coded.
- **Feature importance robustness checks**: re-fitting Random Forest without the best point to flag single-point inflation.
- **SVM classifier + log-magnitude regression**: splitting failed regression into sign classification plus log-space regression on log|Y|.
- **Outlier-suggestion filter**: any model whose argmax is a spatial outlier is dropped from the ensemble.
- **Boundary-consensus rule (refined)**: when 3+ non-linear models push a dimension to its edge AND the correlation sign matches, clip to max(top-K extremum, ensemble interior). Safety without freezing dimensions where valid models support a step.
- **NN gradient analysis** via `torch.autograd` for directional signals at the current best.

**Strategy selection per function:**
A dominant model with strong LOOCV improvement is trusted if interior. Multiple models with mixed agreement get a hybrid: top-K Y-weighted centroid on uncertain dimensions, ensemble where models agree. No model beats baseline → balanced Voronoi space-filling.

**Exploration vs exploitation:**
Calibrated per function by validation performance. Strong-model functions get exploitation; sparse-baseline functions get informed exploration combining a sign classifier on Y, a log-magnitude regressor on log|Y|, and Voronoi targeting of undersampled regions.

**Key learnings after 6 rounds:**
- Linear models extrapolate to boundary corners and are systematically filtered.
- Single outliers inflate correlations and importances; robustness checks come first.
- Some peaks are so narrow that tiny perturbations cause large drops.
- Kernel smoothness is per-function: rougher Matern (ν=0.5) wins on most 4–6D functions; smoother kernels win on the 8D function where the landscape itself is smoother.
- The boundary-consensus rule self-corrects across weeks AND respects interior model agreement (refined this week).
- NN surrogates rarely top the leaderboard at these sample sizes; their main value is the autograd gradient as a directional hint.
