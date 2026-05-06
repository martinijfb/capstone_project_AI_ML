# Week 07 Queries — Submitted 2026-05-05

## Formatted Queries

```
Function 1: 0.273645-0.308963
Function 2: 0.703084-0.943835
Function 3: 0.557343-0.602957-0.175087
Function 4: 0.368531-0.401253-0.431517-0.422540
Function 5: 0.384873-0.966112-0.986585-0.953170
Function 6: 0.426310-0.345626-0.556770-0.734354-0.049005
Function 7: 0.031697-0.473010-0.164637-0.217860-0.330911-0.883237
Function 8: 0.162411-0.233951-0.063190-0.248712-0.707848-0.740771-0.216942-0.589651
```

## Methods

| F | Method | W6 Result | Predicted / Note |
|---|--------|-----------|-------------------|
| F1 | Balanced Voronoi at Q3 (global winner, pre-planned in W6 suggestions) | -1.81e-57 (vanishing negative) | Combined classifier+log-SVR ran but failed 2 trust checks: classifier LOO dropped 87%→81%, candidate 0.045 from a negative. Q3 has 3 negatives + 1 positive — best for negative-region characterisation. d_to_neg ≥ 0.21 (safe). |
| F2 | RMSE-weighted ensemble of 3 interior models (KNN/RF/GB) — STRONG consensus both dims | 0.6961 (NEW BEST, +5%) | Step 0.003 from current best — smallest step of the project. **NEW: BoTorch SingleTaskGP barely beats baseline (+12%)** where sklearn GPs all fail (-7%). GP-UCB and qLogNEI both push x2 to corner — framework rejects. |
| F3 | RMSE-weighted ensemble of RF + GB + Warped-GB + Warped-RF — STRONG x1/x2, moderate x3 (centroid resolves) | -0.0500 (regression, plateau-boundary) | **NEW: Warped-GB +48% beats baseline** (Yeo-Johnson works on F3 unlike F1). GB switched argmax direction to low-x3 cluster (0.689, 0.603, 0.073) we anticipated. Big x3 push 0.340→0.175. |
| F4 | RMSE-weighted ensemble of 7 interior models — STRONG all 4 dims after NN outlier-filter | 0.5524 (NEW BEST, +2%) | Step 0.008 — smallest in F4 history (W4=0.048, W5=0.028, W6=0.028, W7=0.008). **GP-Matern15 +75% leads**. BoTorch generators predict DOWN (0.20-0.24 vs current 0.55) — correctly rejected for F4's narrow peak. |
| F5 | **★ TuRBO-1 with q=1** (deliberate framework deviation; principled BO from research doc) | 2669.31 (NEW BEST, +15.7%) | Standard ensemble step 0.003 too conservative for climbing trajectory. TuRBO Thompson-sampled candidate within ARD-stretched trust region centred on best. Step 0.058. State persisted to `data/function_5/turbo_state.json`. |
| F6 | RMSE-weighted ensemble of 7 interior models — STRONG consensus all 5 dims | -0.1173 (NEW BEST, +55%) | W6 was a +55% leap; this is consolidation week. 4 valid models suggest identical point. **SVR +63% leads**, Matern05 +57%. TuRBO suggests bigger step 0.186 — logged as W8 contingency. |
| F7 | Hybrid pull-back — boundary-consensus on x1, ensemble on x3/x6 (STRONG), top-4 centroid on weak/moderate dims | 1.4147 (regression, -12% from 1.608) | W6's x3=0.143 was too aggressive; pull back to x3=0.165. **GP-Matern05 +60% leads** (4th consecutive function). Notable disagreement: standard ensemble pushes x3 LOW, BoTorch generators push x3 HIGH. corr(x3,Y)=-0.075 (lowest). |
| F8 | Hybrid — ensemble on x1-x5/x7/x8 (STRONG, 7 dims), top-4 centroid on x6 (moderate) | 9.9112 (NEW BEST, +0.04) | **All 10 models beat baseline** (2nd consecutive week). SVR +81%, Matern25/15/RBF all >73%. F8 still favours smoother kernels (opposite of F4-F7 pattern). Continues W6 x5 push direction. TuRBO/UCB/EI all push x5 to 1.0 — logged as W8 contingency. |

## New Pipeline Tools Used This Week

This was the W7 commitment to test the documented pipeline upgrades from `suggestions/pipeline_upgrades_research.md`. Each tool was evaluated:

**TuRBO-1 with q=1:** used as PRIMARY query on F5 (deliberate framework deviation). Logged as W8 contingency for F6/F7/F8. F5 is the load-bearing experiment for whether TuRBO has a permanent role in the pipeline.

**BoTorch SingleTaskGP:** ran as informational LOOCV on F2. Beat baseline at +12% where sklearn GPs all failed (-7%). Useful diagnostic when sklearn's `GaussianProcessRegressor` collapses to mean prediction.

**GP-UCB (β-decay) and qLogNEI:** ran as second-opinion candidate generators on F2/F4/F5/F6/F7/F8. Most produced boundary-pushing suggestions correctly rejected by the framework. On F7 they suggested x3 HIGH while ensemble said x3 LOW — useful as an "alternative hypothesis" signal, since corr(x3,Y) is genuinely ambiguous.

**WarpedRegressor (Yeo-Johnson):** tested on F1 and F3. Failed on F1 (Y range spans 100+ orders of magnitude — Yeo-Johnson can't fit). **Worked on F3** with Warped-GB +48% and Warped-RF +14% — both join the F3 ensemble.

## New Pattern Observed This Week

**Per-function decisions on whether to switch to new tools:** the framework now has a clear discipline rule. Switch to a new tool as primary query only when:
1. Standard pipeline gives a step < 0.005 AND trajectory hasn't saturated → use TuRBO (F5 met this)
2. Two consecutive regressions on a function → try BoTorch alternative direction (F7 is one regression away)
3. All sklearn GPs fail baseline → add BoTorch GP to model list (F1, F2 — already happening)

This week only F5 met the trigger; all others stayed in the standard pipeline. The W7 results will tell us whether F5's TuRBO bet pays off.

## Notable W6→W7 Model Shifts

- **F1 classifier regressed**: 87% → 81% LOO accuracy. The vanishing-negative W6 point at (0.617, 0.222) Y=-1.81e-57 added noise rather than signal to the SVC boundary.
- **F3 NN now beats baseline +6.9% in W7 NN training but fails on the cleaner no-outlier baseline**. Same data-conditioning issue as W6.
- **F8 all 10 models beat baseline for 2nd consecutive week** — the most cleanly fitted function in the project at this point.

## Neural Network Surrogates

Reused W6 NN models from `models/week_06/` (didn't retrain for W7). NN gradients used as directional signals at current best for each function, even when NN itself is excluded from the ensemble due to outlier-filter or boundary rejection.

## Notes

- Date: 2026-05-05
- Total cells: 70 in `notebooks/week_07.ipynb`
- All queries computed from data + models, no hardcoded values
- Plots saved to `plots/week_07/` (per-dim scatter+box, parallel coords, F1 combined+Voronoi, F2 2D scatter)
- Running best Y per function: F1=3.65e-7, F2=0.696, F3=-0.035, F4=0.552, F5=2669, F6=-0.117, F7=1.608, F8=9.911
- W6 produced 5 new bests (F2, F4, F5, F6, F8) and 2 regressions (F3, F7); F1 stayed at zero.
- W7 strategy: continue exploitation where W6 worked (F2/F4/F6/F8); pivot to TuRBO on F5 (climbing + peer gap); pull-back on F7 (W6 overshot); explore low-x3 on F3 (W6 plateau); Q3 Voronoi on F1 (combined approach failed).
- W7 is the load-bearing test for whether the new pipeline tools (TuRBO especially) have a permanent role. Decision deferred to W8 once results land.

## Sources Referenced This Week

- TuRBO: Eriksson et al. (NeurIPS 2019). Implementation in `src/turbo.py`, BoTorch tutorial-style state machine.
- BoTorch SingleTaskGP + UCB / qLogNEI: Balandat et al. (NeurIPS 2020). Wrapped in `src/botorch_helpers.py`.
- WarpedRegressor: HEBO-style Yeo-Johnson on Y (Cowen-Rivers et al. 2022, NeurIPS 2020 BBO Challenge winner). Implementation in `src/output_warping.py`.
- All three documented in `suggestions/pipeline_upgrades_research.md` from end of W6.
