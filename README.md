# Black-Box Optimisation Capstone Project

## Transparency documents

- [Datasheet for the BBO data set](docs/datasheet.md)
- [Model card for the optimisation approach](docs/model_card.md)

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

My approach centres on a **validate-then-trust** framework: fit multiple surrogate models, validate each via Leave-One-Out Cross-Validation (LOOCV RMSE), and only trust models that beat the standard-deviation baseline.

**ML methods used:**
- **GridSearchCV with LOOCV** across 7 sklearn families: Ridge, KNN, Random Forest, SVR (RBF), Gradient Boosting, Gaussian Process with Matern at ν ∈ {0.5, 1.5, 2.5} and RBF, plus PyTorch MLP surrogates. ARD (per-dimension lengthscale) GP variants added for local refinement.
- **Structural transforms**: a Gaussian-magnitude model `f = h(x)·exp(quadratic)` for the sign-flipping function, and a ceiling transform `ln(C−Y)` for the function that saturates toward a cap. Both expose shape the raw-Y models miss.
- **Output warping (Yeo-Johnson)**, **BoTorch second opinions**, **noise-aware GPs (WhiteKernel)** for the one stochastic function, and an **F1 classifier + log-SVR** decomposition.
- **TuRBO-1 trust region (multi-kernel TS)** for still-climbing functions.

**Strategy selection per function:**
Each function is read by its landscape shape:
- **Climbing**: TuRBO continuation (multi-kernel TS picks a different kernel per week).
- **Single peak / pit / plateau**: multi-GP local consensus — average several kernel variants' argmaxes inside a trust radius matched to that function's measured tolerance, anchoring dimensions where the data is unambiguous.
- **Saturated refinement**: Expected Improvement to redirect toward unexplored neighbouring regions.

**Key learnings after 11 rounds:**
- Functions are deterministic except one, which is stochastic (σ≈0.06) — confirmed by exact-repeat queries; that one needs noise-aware GPs and a P(beat) acquisition.
- TuRBO is the right tool only when its step scale matches the landscape. On a unimodal pit (output falling with distance from the best) its large steps are guaranteed losses, so it was retired there for local consensus.
- Single-model dominance is never enough; consensus across kernel variants inside a safe radius is the standing gate.
- Structural transforms (log-magnitude, ceiling) can reveal exploitable shape that raw-Y regression flattens out.
- Distance is the unifying idea: choosing a query is a clustering decision about which region a candidate belongs to.
