# Week 09 Queries — Submitted 2026-05-31

## Formatted Queries

```
Function 1: 0.710746-0.699736
Function 2: 0.703636-0.946935
Function 3: 0.668144-0.679950-0.089905
Function 4: 0.384090-0.400987-0.436714-0.407276
Function 5: 0.920023-0.965586-0.999749-0.959306
Function 6: 0.463763-0.406953-0.414666-0.726397-0.113004
Function 7: 0.098649-0.386696-0.526631-0.173887-0.341660-0.776059
Function 8: 0.089109-0.206213-0.063267-0.243431-0.696063-0.740892-0.194481-0.662394
```

## Methods

| F | Method | W8 Result | Predicted / Note |
|---|--------|-----------|-------------------|
| F1 | Branch 5 maxima-seeker ensemble (KNN + GP-Matern05/15, valley-trackers filtered) | +1.13e-10 (positive, not best) | W8 gradient-climb to (0.724, 0.702) overshot the narrow positive ridge near W3 best (Y dropped 3.65e-7 → 1.1e-10). The 3 maxima-seekers cluster at (0.700-0.724, 0.697-0.702); mean is (0.711, 0.700) — refines between W3 and W8. SVC classifier 83.33% LOO (up from 82.35% W8), still 1.7pp below 85% gate. |
| F2 | **Deliberate noise-test: repeat exact W6 best** | 0.4588 (regression -34%) | 6 queries in a box < 0.04/dim around (0.70, 0.95) span Y ∈ [0.459, 0.696] (std=0.085). W6 (0.7036, 0.9469)=0.6961 and W8 (0.7081, 0.9464)=0.4588 are |Δx|=0.0045 apart yet |ΔY|=0.24 — pure noise signature. Best model RMSE ≈0.13 ≈ cluster noise: models literally cannot resolve the peak interior. Re-querying the exact best gives a second sample to denoise. |
| F3 | RMSE-weighted ensemble of 8 interior models (SVR dropped by outlier filter, x2=0.27 vs cluster at 0.61-0.78) | -0.0274 (NEW BEST, +21%) | **W8 cluster B refinement vindicated** by a major new best. W9 outlier-cleaned analysis has 8/8 valid models pointing to cluster B unanimously. All 3 dims STRONG consensus (spreads 0.07/0.17/0.06). GP-Matern25 predicts a further new best at Y=-0.0249. W9 NN failed baseline (cluster A/B confusion) — gradient ignored. |
| F4 | RMSE-weighted ensemble of 7 models (GB dropped by outlier filter) | 0.1710 (regression -69%) | **Pull back from W8's failed GP-Matern15 single-model trust** (49.9% dominance margin → -69% Y drop). 6 of 8 valid models converge on the EXACT same point — much stronger signal than any single model. All 4 dims STRONG consensus. Step |Δ|=0.020 is half of W8's failed 0.036 step; direction REVERSES W8 on x1 (+0.015 vs -0.016) and x4 (-0.010 vs +0.029) — the dims that caused the cliff. |
| F5 | **★ TuRBO-1 multi-kernel TS** continuation (Branch 6) | 3581.23 (NEW BEST, +6.4%) | 8 consecutive new bests now: 984 → 1207 → 1412 → 1979 → 2308 → 2669 → 3365 → 3581. State after applying W8 success: L=0.4, succ=1, fail=0. Multi-kernel TS picked **RBF kernel** (first time RBF wins F5). TR went full-width on x1 (long lengthscale) → TS draws exploratory x1=0.920, while x2/x3/x4 stay essentially at W8 values. Deliberate x1 exploration beyond the established trajectory plateau. |
| F6 | **★ TuRBO-1 multi-kernel TS (Branch 6 — 2-regression rule triggered)** | -0.2615 (3rd consecutive regression) | **First TuRBO use on F6.** Trajectory: W6 (-0.117) → W7 (-0.178) → W8 (-0.262). Fresh state init: L=0.8, fail=1 after applying W8 result. Multi-kernel TS picked **Matern05** (matches F6 sklearn CV top kernel at +75%). Step |Δ|≈0.15 with all 5 directions matching per-dim correlation signs (x2 down, x3 up, x4 up). |
| F7 | **★ TuRBO-1 multi-kernel TS** continuation (Branch 6) | 2.3305 (NEW BEST, +45%, regression streak broken) | W8 was the framework-mandated TuRBO that **broke a 2-week regression streak with a +45% jump** (1.61 → 2.33). State recorded: fail dropped 4 → 0, succ=1, best=2.33. Multi-kernel TS picked **Matern25**. Step |Δ|≈0.16 explores around the new breakthrough (x1: +0.05, x3: +0.19), trying to find shape now that the TR widened. |
| F8 | Branch 4 per-dim hybrid (STRONG dims = ensemble, weak/moderate dims = top-4 centroid) | 9.7649 (regression -1.5%, fell off plateau) | **W8 deliberate TuRBO plateau-break fell off as cautioned.** Per the W9 contingency: switch back to hybrid to re-anchor. STRONG dims x1-x5 (spreads 0.05-0.19) use ensemble; weak dims x6/x7/x8 (spreads 0.25-0.35) use top-4 centroid. Step |Δ|=0.091 — 5× smaller than W8's TuRBO bet (|Δ|=0.47). Smooth kernels still dominate (Matern25 +92%, RBF +90%). |

## Strategic theme this week

**3 TuRBO bets, 2 deliberate deviations, 3 standard pipeline calls** — the multi-kernel TuRBO has now become a regular tool, not just F5's experiment:
- **F5**: 8 consecutive new bests on TuRBO; multi-kernel TS picked RBF this time (first), Matern15 in W8, single-kernel in W7. Different kernels winning each week proves the multi-kernel ensemble is doing real work.
- **F6**: first TuRBO use, framework-mandated by 3-regression streak. Standard ensemble W7 and W8 both pushed near W6 best and regressed — TuRBO breaks the loop.
- **F7**: continuation after a +45% breakthrough. State recorded fail 4 → 0 + succ=1.

**Two deliberate deviations this week (F1 and F2)**:
- F1: maxima-seeker ensemble after gradient-climb overshoot. The "filter valley-trackers" idea from W8 is now standard for F1 until the classifier crosses 85%.
- F2: noise-test repeat of W6 best. Spending 1 query to quantify noise (σ≈0.085 in the near-peak cluster) is more valuable than another small-step refinement that can't distinguish signal from noise.

**Two pull-backs from W8 missteps**:
- F4: ensemble pulled back from single-model trust (W8 GP-Matern15 with 49.9% margin still failed -69%).
- F8: hybrid pulled back from TuRBO plateau-break (W8 deliberate bet fell off as I warned).

**Lessons being applied**:
- Outlier-correlation check now standard on every F3 analysis (W8 F3 catch).
- Single-model dominance margins (even +91% on F4) are not enough — multi-model consensus is more reliable.
- TuRBO is the right answer when the trigger conditions are met; ignoring it costs us regression weeks.

## TuRBO state files updated

- `data/function_5/turbo_state.json`: succ=1, fail=0, best=3581.23 (after applying W8 success)
- `data/function_6/turbo_state.json`: succ=0, fail=1, best=-0.117 (after applying W8 regression)
- `data/function_7/turbo_state.json`: succ=1, fail=0, best=2.33 (after applying W8 success, fail 4→0)
- `data/function_8/turbo_state.json`: succ=0, fail=1, best=9.91 (after applying W8 fall-off; F8 switched away from TuRBO this week)

## Notable W8 → W9 model shifts

- **F3 NN dropped from beating baseline to failing** (+6.0% → -12.6%). The cluster A vs cluster B confusion from the new best at (0.66, 0.62, 0.07) makes the NN unable to reconcile two competing clusters. NN gradient ignored for F3.
- **F7 NN improvement halved** (+42.6% → +18.5%). The W8 Y=2.33 point is a sharp jump above the previous cluster max of 1.61; the NN smooths over this and loses fit quality.
- **F1 NN classifier** 70.6% → 77.8% (up). SVC classifier 82.35% → 83.33% (closer to 85% gate; trajectory suggests crossing within 1-2 weeks).
- **F4: GP-Matern15 still leads CV at +91%** but we explicitly do NOT trust single-model dominance after W8's failure. Multi-model consensus is the new floor.

## Notes

- Date: 2026-05-31
- Total cells in `notebooks/week_09.ipynb`: 73 (including plots, models, decisions)
- All queries computed from data + models, no hardcoded values
- Plots saved to `plots/week_09/` (per-dim scatter, parallel coords, F1 2D scatter)
- Running best Y per function: F1=3.65e-7, F2=0.6961, **F3=-0.0274 (W8★)**, F4=0.5524, **F5=3581.23 (W8★)**, F6=-0.1173, **F7=2.3305 (W8★)**, F8=9.9112
- W8 produced 3 new bests (F3, F5, F7), 4 regressions (F2, F4, F6, F8), F1 landed positive but not best
- W9 strategy: continue TuRBO on F5/F7 (validated), start TuRBO on F6 (rule-triggered), pull back to hybrid on F4/F8 (W8 misstep correction), refine on F3 (cluster B confirmed), maxima-seeker on F1 (post-overshoot), noise-test on F2 (cluster σ≈0.085)
- Multi-kernel TuRBO state: 3 different winning kernels this week (RBF for F5, Matern05 for F6, Matern25 for F7) — kernel diversity is genuinely paying off.

## Sources Referenced This Week

- TuRBO multi-kernel TS: Shibata et al. (Optuna BBO Challenge 2020 entry). `src/turbo.py`.
- Output warping (Yeo-Johnson): F3 cluster B recommendation from Warped-GB. `src/output_warping.py`.
- Multi-model consensus over single-model dominance: lesson from F4 W8 failure now baked into Cell E decision logic.
