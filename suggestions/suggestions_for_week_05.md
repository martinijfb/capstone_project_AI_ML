# Suggestions for Week 05

Collected as Week 04 `/analyze` cells are completed. Update per function.

## Per-Function Recommendations

### F1 (2D, 14 pts after W04 query)
- Week 04 query: **balanced Voronoi at (0.665, 0.437)** — interior Q4, equidistant (0.245) from both big negatives
- Method: `min(dist to data, dist to boundary)` — replaces naive Voronoi which picked the corner (0.05, 0.95). Corners are degenerate for BBO (boundary effects, one-sided info), so we penalize corner-proximity equally with cluster-proximity
- Expected outcome: whatever Y returns, it strongly refines the classifier — point is nearly equidistant from the two big negatives (0.42, 0.46) and (0.65, 0.68)
- Note: classifier already at 92.3% LOO (above 85% threshold), but combined approach was skipped because log-SVR is miscalibrated — it extrapolates high |Y| from the 2 large negatives since all observed positives are ~0. Combined candidate (0.734, 0.681) predicted Y ≈ 1e-11 — no real upside

**IMPORTANT: F1 outliers are SIGNAL, not noise (do NOT remove them, unlike F3):**
- pt10 at (0.4211, 0.4636) Y=-6.63e-3  — largest |Y|
- pt4 at (0.6501, 0.6815) Y=-3.61e-3  — 2nd largest |Y|
- All 11 other points have |Y| < 1e-15 (essentially 0 at machine precision)
- Removing these would drop baseline RMSE 100× and kill the classifier / log-SVR — they ARE the training data that defines the sign boundary and magnitude
- Contrast with F3 where the outlier was distorting x3 correlations. F1's outliers define the problem structure
- Cell B plots circle them in red so the reason is visually obvious

- **If Y is positive (≈ 0)**: Q4 is mostly positive → classifier boundary tightens around the negatives; target Q2 (0.229, 0.566) next week
- **If Y is large negative (< -1e-3)**: negative region extends further → log-SVR gets a 3rd negative anchor (finally real magnitude diversity!), Q2 balanced Voronoi next week with classifier retrained
- **If Y is LARGE positive (> 0.01) — EXIT CONDITION**: switch to exploitation around (0.665, 0.437); we found the peak region
- Long-term strategy: `suggestions/f1_long_term_strategy.md` — continue balanced space-filling through ~W07, then return to classifier + log-SVR combined approach
- For the combined approach to become useful, we need at least one positive observation with non-trivial magnitude — currently all 9 positives are numerically zero

### F8 (8D, 44 pts after W04 query)
- Week 04 query: **(0.0861, 0.2164, 0.0229, 0.1329, 0.5005, 0.7394, 0.1501, 0.7281)** — hybrid with **boundary-consensus override on x3**
- Decision branch: #5 + boundary-consensus clip
- **Boundary-consensus rule (new this week)**: 4 non-Ridge models pushed x3 below 0.02 (KNN 0.01, GB 0.002, GP-Matern 0.002, Ridge counted separately). With r(x3)=-0.687 (strongest) and pt#4 at x3=0.023 Y=9.60, boundary signal is real — clip x3 to top-5 min (0.023) instead of extrapolating to 0. x1 had only 2 non-Ridge models agreeing (below the ≥3 threshold), so it stays at ensemble 0.086.
- Per-dim sources: x3→boundary-consensus; x1/x4/x7→ensemble (STRONG interior); x2/x5/x6/x8→Y-weighted top-4 centroid (weak/moderate)
- Model fit (43 pts): all 8 beat baseline strongly. GP-Matern +83%, SVR +79%, GP-RBF +78%, NN +62%. 6 of 8 hit boundaries — models are aggressively extrapolating, but the per-dim consensus filter catches which boundaries are real (x3) vs which are artifacts (x1, x4, x7).
- Distance from current best: 0.151 — moderate step
- Y-history: W1-W2 climb (9.50 → 9.87), W3 plateau (9.82), W4 tests boundary-informed direction
- **If Y > 9.87 (new best)**: boundary-consensus approach validated — next week, if ≥3 non-Ridge models still push x3 low, keep x3 at 0.023 or lower. Check if other dims develop boundary consensus as data grows.
- **If Y ~ 9.5-9.87 (still near plateau)**: low x3 didn't unlock more. Try pure exploitation: small perturbation from current best (step=0.03 in correlation direction).
- **If Y < 9.5 (overshoot)**: clipping x3 went too far. Pull back x3 to 0.05 (between best's 0.08 and #4's 0.023).
- With 44 pts: consider retraining NN next week and re-checking boundary-consensus pattern — if more dims develop consensus, the boundary-informed approach may work even better.

### F7 (6D, 34 pts after W04 query)
- Week 04 query: **(0.0644, 0.4815, 0.2432, 0.2299, 0.3414, 0.8121)** — hybrid: ensemble on STRONG-consensus dims (x1, x3, x4, x5, x6) + Y-weighted centroid of top 4 on weak dim (x2)
- Decision branch: #5 (STRONG on 5/6 dims, weak only on x2 where models disagree: 3 models at 0.54, RF at 0.93, SVR at 0.42)
- Continued the successful hybrid approach from W3 — F7 climbed W2(1.125) → W3(1.365) → W4(1.461) via this exact pattern
- Model fit: all 8 beat baseline. GP-Matern +47%, GP-RBF +45%, KNN +43%, SVR +40%, NN only +13%. Ridge/GB/NN boundary-excluded.
- Distance from best: 0.063 — consistent with prior successful steps
- NN gradient at best: check cell E output for dim-specific push direction
- **If Y > 1.461 (new best)**: hybrid working — continue same approach. Re-run hybrid next week with 34 pts (x2 consensus may tighten as data grows).
- **If Y similar (1.2-1.4)**: plateau near peak — try pure ensemble (no hybrid) to see if x2=0.542 (where 3 models agreed) is actually better than centroid's 0.48.
- **If Y drops (< 1.2)**: hybrid overshot — pull back with smaller step (weighted 0.5*hybrid + 0.5*current_best).
- Watch: x1 consistently near 0.04-0.06, very low. x5 drifting down (0.37 → 0.34). If x5 hits boundary next, pull back.

### F6 (5D, 24 pts after W04 query)
- Week 04 query: **(0.4237, 0.4370, 0.5013, 0.8059, 0.0467)** — RMSE-weighted ensemble of KNN/RF/GB/GP-Matern/GP-RBF (5 models)
- Decision branch: #5 (STRONG consensus on x2/x4/x5, moderate on x1/x3; GP-Matern only 1.09× dominant — not enough to solo)
- **SVR lost its dominance this week**: in W3 SVR dominated (55%), this week SVR is 3rd-best (+58.9%) AND hit boundary (x5=0.005) → excluded. With 23 pts + more model diversity, SVR-only reliance is ending.
- **NN also boundary-rejected**: suggests Y=+0.67 at (0.21, 0.03, 0.13, 0.37, 0.001) — wild extrapolation far above any observed Y (max -0.31). Not trustworthy.
- ALL 8 families beat baseline. GP-Matern +44%, GP-RBF +39%, NN +37%, KNN +33%, GB +33%, RF +34%.
- Trajectory: W1 pt3 (-0.61) → W2 (-0.44) → W3 (-0.31). Consistent +0.13-0.17/week climb via SVR in the past.
- Step size 0.088 from current best — similar to prior weeks
- x5 dims consistently drives low (~0.04-0.06 across all top picks) — STRONG signal, r(x5)=-0.66
- **If Y > -0.3 (new best, ≥+0.01 improvement)**: ensemble is calibrated — continue with same approach. Expect Y approach 0.
- **If Y ~ -0.4 to -0.3 (plateau)**: near local peak — try GP-Matern's pick (0.403, 0.480, 0.480, 0.767, 0.060) as alternative.
- **If Y < -0.5 (overshoot)**: pull back to W3 best + small perturbation: (0.40, 0.38, 0.50, 0.85, 0.05).

### F5 (4D, 24 pts after W04 query)
- Week 04 query: **(0.3113, 0.9083, 0.9479, 0.8983)** — RMSE-weighted ensemble of 6 interior models (KNN/RF/SVR/GB/GP-Matern/GP-RBF)
- Decision branch: #5 (STRONG consensus on all 4 dims; GP-Matern marginally dominant at 1.60×)
- **F5 is the most reliably-climbing function**: W1=984 → W2=1207 → W3=1412 (+200 per week, all via ensemble method)
- W4 step from W3: 0.065 distance — similar magnitude to prior steps that kept climbing
- ALL 8 model families beat baseline this week. GP-Matern (+78%), NN (+75%), KNN (+65%), RF (+56%), GB (+51%), GP-RBF (+54%), Ridge (+43%), SVR (+29%). NN hit boundary, Ridge is linear extrapolator — both excluded.
- NN gradient at best: large positive on all dims (especially x3, x4) — consistent with climbing direction
- **If Y > 1412 (new best, expected)**: ensemble is calibrated — continue same approach, re-fit with 24 pts. Ensemble direction predicts +1555-1693.
- **If Y ~ 1200-1412 (plateau)**: near peak — try smaller perturbation: midpoint of W4 query and current best, step ~0.03.
- **If Y < 1000 (overshoot)**: ensemble got too aggressive — try GP-Matern's dominant pick (0.365, 0.924, 0.967, 0.892) as alternative.
- Trajectory note: every week's ensemble has pushed x2/x3/x4 up toward 1.0 — watch for boundary saturation. Current x3=0.948 is closest to 0.98 safety margin; next week may need to cap at 0.95.

### F4 (4D, 34 pts after W04 query)
- Week 04 query: **(0.3696, 0.4039, 0.4105, 0.4312)** — RMSE-weighted ensemble of 6 models (KNN/RF/SVR/GB/GP-Matern/GP-RBF); NN excluded as spatial outlier (suggested (0.02, 0.51, 0.54, 0.56) — far from consensus cluster)
- Decision branch: #5 (STRONG consensus on all 4 dims after exclusions)
- Model fit: GP-Matern dominant (+75.8%), SVR (+71.3%), NN (+48.8%), 7 models total beat baseline. GP-Matern/SVR ratio 1.18× → comparable (not dominant).
- Spread per dim (after outlier exclusion): x1=0.18, x2=0.06, x3=0.05, x4=0.11 — STRONG everywhere
- Distance from current best (0.385, 0.429, 0.410, 0.393): 0.048 — smaller than W3's 0.07 (which caused -0.03) and W2's 0.07 (which caused -1.39)
- F4 has a very sharp peak: +0.37 best, but small moves drop Y rapidly. Ensemble is roughly the midpoint between W3 query and best (as prior suggestion recommended).
- NN gradient at best: dx1=-8.6 (big), dx2=+3.4, dx3=+0.1, dx4=+6.0 → direction agrees with ensemble (x1↓, x2↑/neutral, x4↑)
- **If Y > 0.37 (new best)**: ensemble is calibrated — continue same approach with 34 pts. Expected Y increase into positive territory.
- **If Y ~ 0 to 0.37**: models are close but not converging to peak. Take midpoint between this ensemble and current best (smaller step ~0.024).
- **If Y < 0 (worse than W3)**: ensemble overshot. Pull back halfway between ensemble and W1 best (0.377, 0.416, 0.410, 0.412) with step ~0.02.
- **If Y << -1 (catastrophic like W2)**: the 4D peak is too narrow — use only GP-Matern's direct suggestion (0.340, 0.385, 0.407, 0.427) as pure exploitation.
- Watch: function sensitivity means every W5 step should be ≤0.04 per dim.

### F3 (3D, 19 pts after W04 query)
- Week 04 query: **(0.3679, 0.4565, 0.3375)** — GB dominant model's suggestion (GB beats baseline by +55% after outlier removal)
- **Critical outlier finding**: pt at (0.15, 0.44, 0.99) Y=-0.40 was masking the signal. With it: r(x3,Y)=-0.58 (dominant), 0 models beat baseline. Without it: r(x3,Y)=-0.17 (weak), GB beats by +55%, RF by +6%. This is exactly the outlier-influence pattern flagged in memory.
- **ALWAYS analyse F3 both with AND without this outlier** — it's a single point with Y-gap of 0.27 to the 2nd-worst, clearly separated.
- Decision branch: #2 (one model dominates) — GB has 8.9× lower RMSE than RF runner-up. GB suggestion is interior (not boundary) and predicts Y matching current best.
- True correlations (no outlier): r(x1)=+0.02, r(x2)=+0.14, r(x3)=-0.17 — ALL WEAK. Bimodal hypothesis still plausible (top-2 at x3=0.34 vs x3=0.066).
- **If Y improves (> -0.035)**: GB is calibrated — continue with GB suggestion next week, re-fit with 19 pts.
- **If Y similar (-0.05 to -0.035)**: GB identified a new local plateau — try ensemble direction (0.443, 0.562, 0.254) as alternative.
- **If Y worse (-0.1 to -0.05)**: GB overfit — try RF's alternative suggestion (0.601, 0.781, 0.079) [the other mode].
- **If Y much worse (< -0.1)**: both models wrong direction — pull back to pure exploitation of current best with step=0.03.
- NN (ensemble/H32) was trained WITH outlier so its RMSE is baseline-comparable only. Consider retraining NN without outlier in W5 `/train-nns`.

### F2 (2D, 14 pts after W04 query)
- Week 04 query: **(0.8958, 0.1240)** — interior-safe exploration of the "low-x2 ridge" hypothesis (deferred the safe ensemble at (0.69, 0.96) this week)
- Rationale: pt#3 at (0.67, 0.12) Y=+0.54 is an anomaly — the 3rd-best Y at LOW x2 when all other top-5 points have x2>0.5. SVR's boundary pick at (0.998, 0.000) was the clue but extrapolated to a degenerate corner. (0.90, 0.12) tests the same direction while staying interior.
- W4 model fit: 7/8 beat baseline, STRONG consensus (spread 0.03×0.02) among interior models (KNN/RF/GP-Matern/GP-RBF) on the safe ensemble target. Deferred.
- NN gradient at best: dx2=+2.31, dx1=-0.37 — supports the ensemble direction (pushing x2 up), NOT the risk direction. If the risk returns low Y, this gradient info confirms to return to exploit mode.
- **If Y > 0.5 at (0.90, 0.12)**: low-x2 ridge is REAL → next week exploit around here. Try (0.90, 0.15) or (0.85, 0.10) perturbations. May have found a second/bigger peak.
- **If Y ~ 0.3–0.5 at (0.90, 0.12)**: ridge exists but not higher than current best. Return to the ensemble direction: (0.695, 0.960) perturbation next week.
- **If Y < 0.3 at (0.90, 0.12)**: no ridge, pt#3 was anomalous (possibly noise). Fall back to deferred safe ensemble (0.69, 0.96) next week.
- **If Y < 0** (negative): region is flatly bad — confirms high-x2 cluster is the true peak region; exploit with smaller perturbation (0.695, 0.960).
- Previous running best (0.6658 at W3 query (0.702, 0.953)) is preserved for BBO scoring regardless of this week's outcome.
