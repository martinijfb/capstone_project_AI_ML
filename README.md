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
- **GridSearchCV with LOOCV** across 7 sklearn families: Ridge, KNN, Random Forest, SVR (RBF), Gradient Boosting, Gaussian Process with Matern at ν ∈ {0.5, 1.5, 2.5} and RBF, plus PyTorch MLP surrogates.
- **Output warping (Yeo-Johnson)** for skewed but bounded targets; **BoTorch second opinions** (`SingleTaskGP` + GP-UCB + qLogNoisyEI) as informational signals.
- **TuRBO-1 trust region (multi-kernel TS)**: fits four GPs (Matern 0.5/1.5/2.5, RBF), draws Thompson samples at shared candidates, picks argmax across the (kernel, candidate) grid. Trust-region length adapts via success/failure counters; state persists across weeks via JSON.
- **F1-specific classifier + log-SVR** decomposition. **Feature importance robustness, outlier-suggestion filter, boundary-consensus rule, NN autograd gradients** continue from earlier weeks.

**Strategy selection per function:**
By week 10 the eight functions cluster into three regimes:
- **Climbing**: TuRBO continuation. Multi-kernel TS picks a different winning kernel per function and per week.
- **Models converged**: per-dim hybrid — ensemble where models agree (STRONG-consensus dims), top-K centroid where they disagree, with deterministic anchors where the data shows one.
- **Stalled**: smallest available step around the current best, leaning on cross-model consensus rather than any single confident pick.

Triggers for switching to TuRBO: standard step < 0.005 on a climbing trajectory, or 2 consecutive regressions.

**Key learnings after 10 rounds:**
- Single-model dominance is not sufficient even at very high CV margin; multi-model consensus is the standing gate.
- Kernel smoothness is per-function. Multi-kernel TS earns its cost because the winning kernel changes by function and by week.
- Transform research on F1 found `log|Y|` fits +68% above baseline vs raw Y at +47% — magnitude is smooth, sign is chaotic. Signed-log extrapolation produces nonsense argmax candidates.
- A deliberate small-step query near the current best resolves "refinement vs noise" ambiguity for the entire downstream search.
- Output warping helps where Y is skewed but bounded; useless on extreme dynamic ranges.
- NN surrogates rarely top the leaderboard at these sample sizes; their main value is the autograd gradient as a directional hint.
- TuRBO's value is asymmetric: clear payoffs when real structure is there to find, but it can fall off when the function is genuinely flat. The state machine contracts the trust region after failures so the next bet is cheaper.
