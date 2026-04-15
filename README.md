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

My approach centres on a **validate-then-trust** framework: fit multiple surrogate models, validate each with Leave-One-Out Cross-Validation (LOOCV RMSE), and only trust predictions from models that demonstrably beat the baseline (predicting the mean).

**ML methods used:**
- **GridSearchCV** with LOOCV across 7 model families: Ridge, KNN, Random Forest, SVR (RBF), Gradient Boosting, Gaussian Process (Matern and RBF), and PyTorch MLP surrogates with regularisation variants (plain / dropout / weight-decay / ensemble)
- **Feature importance robustness checks**: re-running Random Forest importance without the best point or known outliers to detect single-point inflation
- **SVM classification + log-magnitude regression**: splitting failed regression problems into sign classification plus log-space regression to identify promising regions
- **Model convergence analysis**: measuring per-dimension spread of model suggestions to assess consensus
- **NN gradient analysis** via `torch.autograd` for directional signals at the current best point

**Strategy selection per function:**
The strategy adapts based on model reliability. When a dominant model achieves strong LOOCV improvement, I trust its suggestion if it stays interior. When multiple models beat baseline but disagree on some dimensions, I use a hybrid: centroid of top performers on uncertain dimensions, model consensus where all models agree. When no model beats baseline, I fall back to Y-weighted centroids or balanced Voronoi space-filling.

**Exploration vs exploitation:**
The balance is calibrated per function by validation performance. Strong-model functions get exploitation; weak-model functions get exploration via balanced Voronoi (which penalises both cluster-adjacency and boundary-proximity, avoiding the corner-picks that naive Voronoi produces). The principle: exploitation only makes sense where models have earned trust through cross-validation.

**Key learnings after 4 rounds:**
- Linear models extrapolate to boundary corners — systematically filtered from ensembles
- Single outliers can inflate correlations and importances — robustness checks before trusting any signal
- Function sensitivity varies enormously — some peaks are so narrow that tiny perturbations cause large drops
- When multiple non-linear models push the same dimension to a boundary AND the correlation sign agrees, the signal is real — I clip to the observed top-K min/max rather than extrapolate to extremes
- NN surrogates beat baseline on most functions but rarely top the leaderboard at these sample sizes — their main value is gradient analysis, not prediction
