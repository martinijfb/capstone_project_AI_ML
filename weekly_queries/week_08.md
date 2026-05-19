# Week 08 Queries — Submitted 2026-05-19

## Formatted Queries

```
Function 1: 0.724297-0.702040
Function 2: 0.708074-0.946424
Function 3: 0.658278-0.616243-0.072861
Function 4: 0.350866-0.417803-0.439220-0.444161
Function 5: 0.385402-0.965623-0.999570-0.959221
Function 6: 0.386326-0.363766-0.545708-0.735494-0.047722
Function 7: 0.052985-0.285792-0.338086-0.214976-0.316571-0.779126
Function 8: 0.008646-0.016274-0.188833-0.352216-0.793490-0.863726-0.146039-0.933019
```

## Methods

| F | Method | W7 Result | Predicted / Note |
|---|--------|-----------|-------------------|
| F1 | **Manual gradient-climb** (deliberate Branch 1 deviation; KNN-validated) | -2.08e-25 (vanishing negative) | After 7 weeks of space-filling, step 0.025 from W3 best (0.700, 0.695) along the +/− gradient direction (away from closest negative at (0.650, 0.682) Y=−3.6e-3). KNN was the only maxima-seeker model (predicted Y > 0); 5 other models were valley-trackers. SVC classifier rose to 82.35% (still under 85% gate). |
| F2 | RMSE-weighted ensemble of 6 interior models (RF, GB, GP-Matern05/15/25, GP-RBF) | 0.5756 (regression -17%) | Branch 4 STRONG consensus all dims (spreads 0.037, 0.035). KNN dropped by outlier filter. NN gradient at best [+0.92, +0.89]. Step +0.0044 from W6 best in x1. W9 contingency: if regression again, repeat W6 best to test noise hypothesis. |
| F3 | **Warped-GB argmax (Branch 2)** on outlier-cleaned data | -0.1237 (regression -250%) | **MAJOR CATCH: F3 outlier at (0.15, 0.44, 0.99) Y=-0.40 was driving r(x3,Y) from -0.55 (artifact) to -0.10 (real).** Without outlier, 2 strongest models (GB +69%, Warped-GB +67%) both point to cluster B refinement (low x3). GB rejected for suspect x2=0.914. Warped-GB at (0.658, 0.616, 0.073) is consistent with 2nd-best init point Y=-0.0364. |
| F4 | **GP-Matern15 argmax (Branch 2)** | 0.5506 (essentially flat -0.3%) | Dominant model at +91.2% RMSE improvement, **49.9% relative gap** over runner-up GP-Matern05 (+82.1%). All 4 dims STRONG consensus. NN gradient at best [-12.8, +10.1, -1.9, +5.0] supports the direction. Step |Δ|=0.036, 3× larger than ensemble — bolder bet to escape W7's flat result. |
| F5 | **★ TuRBO-1 multi-kernel TS** continuation (Branch 6) | 3365.22 (NEW BEST, +26%) | State: L=0.8, succ=1, fail=0, best=3365.22 (W7's success). Multi-kernel TS picked **Matern15** kernel. Trajectory still climbing every week. Step |Δ|=0.018: x3 pushed from 0.987 → 0.9995, x4 +0.012, x1/x2 essentially unchanged. State saved to `data/function_5/turbo_state.json`. |
| F6 | RMSE-weighted ensemble of 7 interior models (Branch 4) | -0.1781 (regression -52%) | All 5 dims STRONG consensus (spreads 0.015-0.136). GB dropped by outlier filter. SVR top at +76%, GP-Matern05 +74%. NN gradient at best [-0.57, -0.57, -1.28, -1.39, -2.04] supports decreasing all dims. Step reverses W7's x1 direction (W7 went +0.006 and regressed; W8 goes -0.034). |
| F7 | **★ TuRBO-1 multi-kernel TS (Branch 6 — 2-regression rule)** | 1.1157 (regression, 2nd in a row from W5 peak 1.6078) | **Framework-mandated TuRBO trigger met** after 2 consecutive regressions: W5 (1.6078) → W6 (-12%) → W7 (-21%). State: L=0.8, succ=0, fail=1. Multi-kernel TS picked **Matern05** (matches sklearn CV's top model GP-Matern05 +68%). Standard ensemble rejected for extrapolating x5=0.194 below top-5 range. |
| F8 | **★ TuRBO-1 multi-kernel TS (Branch 6 — deliberate plateau-break)** | 9.8992 (-0.012 from W6, first sub-best result) | **Deliberate deviation** — F8 hasn't strictly met 2-regression rule but 3 weeks of plateau at ~9.9 with tiny refinement steps. User judgment: "small steps in a plateau won't lead anywhere." Multi-kernel TS picked **Matern05** (deliberately rougher than sklearn CV's Matern25 +91% preference). |

## New This Week

**TuRBO multi-kernel Thompson sampling**: First production use of the multi-kernel upgrade implemented this week (Optuna BBO 2020 paper, Shibata et al.). TuRBO now fits 4 GPs (Matern 0.5/1.5/2.5, RBF) and draws TS from each at shared candidates, picking argmax across the (kernel, candidate) grid. Three F-cells used TuRBO this week (F5, F7, F8) and winning kernels were:
- F5: **Matern15**
- F7: **Matern05** (matches sklearn CV top kernel for F7)
- F8: **Matern05** (deliberately rougher than sklearn CV's Matern25 winner for F8)

**Outlier-correlation check now standard**: After F3's W6 misadventure (smooth-GP extrapolation hypothesis was an artifact of one extreme point), every function's Cell A now prints WITH/WITHOUT correlation table to catch similar issues. F3 fixed mid-analysis after the user flagged this. F4-F8 confirmed clean.

## F1 Deliberate Deviation

After 7 weeks of classifier/Voronoi space-filling with no Y improvement (best stuck at +3.65e-7 from W3), switched to manual gradient-climbing from the only known positive point along the only known sign boundary. KNN's argmax independently validated the W3-best region. Gradient map saved to `plots/week_08/function_1_gradient_map.png`.

## TuRBO Trigger Decisions

This week became a stress-test of the TuRBO branch:
- F5: continuation (W7 succeeded)
- F7: framework-mandated (2 regressions)
- F8: deliberate (user judgment over plateau-stuck hybrid)

W9 will tell us whether multi-kernel TS works on smooth (F8) and plateau-stuck (F7) functions in addition to the validated climbing case (F5).

## Notable W7→W8 Model Shifts

- **F1: 8 of 10 models beat baseline for the first time in 7 weeks** — but the ensemble was almost all valley-trackers; only KNN was a maxima-seeker.
- **F3: outlier removal flipped recommendations** — original analysis suggested smooth-GP extrapolation (mid-x3); cleaned analysis suggests low-x3 cluster B.
- **F4: GP-Matern15 became dominant** at +91.2% with 49.9% margin (previously GP-Matern15 led F5-F7 but not F4 by such a wide margin).
- **F8: Matern25 still leads sklearn CV at +91.1%** — F8 remains the only function consistently preferring smoother kernels. The TuRBO Matern05 selection is the deliberate counterpoint.

## Neural Network Surrogates

W8 NN models trained at start of week, 7/8 beat baseline (F1 the lone failure as usual). F5's NN now at +88.6% improvement (was +12% in W7) thanks to the new (0.385, 0.966, 0.987, 0.953) → 3365 point. F1 classifier dropped to 70.6% (NN-based) but SVC variant is at 82.35% (the relevant one for F1 gate). NN gradients used as directional hints in F2, F3, F4, F6, F8 analyses.

## Notes

- Date: 2026-05-19
- All queries computed from data + models, no hardcoded values
- Plots saved to `plots/week_08/` (per-dim, parallel coords, F1 gradient map, F2 2D scatter)
- Running best Y per function: F1=3.65e-7, F2=0.696, F3=-0.035, F4=0.552, F5=3365.22, F6=-0.117, F7=1.608, F8=9.911
- W7 produced 1 new best (F5 TuRBO, +26%) and 7 regressions of various magnitudes
- W8 strategy: 2 deliberate deviations (F1 gradient-climb, F8 plateau-break TuRBO), 1 framework-mandated TuRBO (F7), 1 TuRBO continuation (F5), 2 STRONG-consensus ensembles (F2, F6), 2 dominant-model picks (F3 Warped-GB, F4 GP-Matern15)
- TuRBO state files updated for F5, F7, F8: `data/function_{5,7,8}/turbo_state.json`

## Sources Referenced This Week

- TuRBO multi-kernel Thompson sampling: Shibata et al. (Optuna team, NeurIPS 2020 BBO Challenge 5th place). Implemented in `src/turbo.py` this week.
- HEBO output warping (Cowen-Rivers et al., NeurIPS 2020 BBO winner): `src/output_warping.py` — Warped-GB drove F3's revised W8 query.
- Hybrid surrogate ensembles (Du Xiaoman, NeurIPS 2020 BBO 4th place): validates our BoTorch+sklearn dual-track approach.
- Papers reviewed end of W7, summary in `suggestions/pipeline_upgrades_research.md`.
