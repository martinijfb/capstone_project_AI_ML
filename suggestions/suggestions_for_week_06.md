# Suggestions for Week 06

Collected as Week 05 `/analyze` cells are completed. Update per function.

## Per-Function Recommendations

### F2 (2D, 15 pts after W05 query)
- Week 05 query: **(0.6939, 0.9626)** — RMSE-weighted ensemble of 5 interior models (KNN/RF/GB/GP-Matern/GP-RBF). Step 0.012 from current best.
- Decision branch: #5 (STRONG consensus on both dims)
- W4 outcome: the risk gambit at (0.90, 0.12) returned Y=0.085 — killed the "low-x2 ridge" hypothesis. Ensemble was the right call after all.
- Model fit (14 pts): 7/8 beat baseline, GB best at +18.7%. NN failed (-0.4%) — the W4 outlier at (0.90, 0.12, Y=0.085) confused MLP fit.
- SVR was a spatial outlier this week (auto-excluded by mean+2σ rule)
- NN gradient at best: dx1=-0.55 (slightly down), dx2=+0.19 (slightly up) — corroborates ensemble direction
- **If W5 Y improves (> 0.6658)**: ensemble is calibrated → continue with same approach next week. Expect plateau as we approach the peak.
- **If W5 Y similar (0.55-0.67)**: peak is very narrow around current best. Try smaller perturbation: midpoint of W5 query and current best.
- **If W5 Y drops (< 0.5)**: the W4 outlier may still be poisoning the fit — try retraining without (0.896, 0.124) excluded as an outlier-candidate, then re-ensemble.
- Watch: x2 still climbing (0.953 → 0.963). If next pick is > 0.97, flag boundary-adjacent risk.

### F8 (8D, 45 pts after W05 query)
- Week 05 query: **(0.0848, 0.2188, 0.0519, 0.1795, 0.5029, 0.7407, 0.1576, 0.7261)** — hybrid: ensemble on STRONG dims (x1, x3, x7), top-4 centroid on weak/moderate (x2, x4, x5, x6, x8). No boundary-consensus this week.
- Decision branch: #5 (STRONG on 3, moderate on 2, weak on 3)
- W4 outcome: boundary-consensus on x3 (clip to 0.023) gave Y=9.85 (slight regression from best 9.87). Boundary push didn't help.
- Boundary-consensus shift: x3 had 4 non-Ridge models pushing low in W4, only 1 in W5. Adding the W4 point at x3=0.023 may have shifted model confidence.
- Model fit (44 pts): all 8 beat baseline. GP-Matern best +84%, SVR/GP-RBF +79%, NN +63%. 4 boundary-rejected.
- GP-Matern dominant 1.29× — comparable, not dominant
- Distance from current best: 0.129 (mostly via x1 reduction 0.191→0.085 and x7 reduction 0.224→0.158, both STRONG-consensus moves)
- NN gradient at best: dx5=+0.99 strongest (push x5 up), dx3=+0.55, dx6=+0.35. Partial disagreement with hybrid (which keeps x5/x3 stable). NN below GP-Matern in accuracy so deferred.
- **If W5 Y > 9.87 (new best)**: hybrid validated — continue with same approach. Aggressive x1/x7 reductions worked.
- **If W5 Y similar (9.5-9.87)**: peak is local plateau — try GP-Matern's solo pick (0.075, 0.270, 0.031, 0.350, 0.735, 0.442, 0.210, 0.332) as a different region.
- **If W5 Y drops (< 9.5)**: hybrid pushed too far — pull back, try midpoint between W5 query and current best.
- Week 05 query: **(0.0542, 0.4678, 0.2205, 0.2157, 0.3166, 0.7794)** — hybrid: ensemble on STRONG dims (x1-x5), Y-weighted top-4 centroid on weak dim (x6)
- Decision branch: #5 (STRONG on 5/6 dims, moderate on x6 spread 0.218; GP-Matern 1.06× — not dominant)
- Continued the successful hybrid approach from W3-W4 — F7 climbing slowly: 1.125 → 1.365 → 1.461 → 1.493
- Model fit (34 pts): all 8 beat baseline. GP-Matern +50.4%, GP-RBF +47.6%, SVR +46.2%, KNN +44.1%. Ridge/GB/NN excluded (boundary).
- The "weak dim" rotated this week: was x2 in W4, now x6 in W5. Same hybrid principle, different dim
- NN gradient at best: dx1=-2.65 (push x1 down hard), dx5=-1.13 (push x5 down) — direction-consistent with ensemble
- Distance from current best: 0.052 — similar to prior successful steps
- **If W5 Y > 1.493 (new best)**: hybrid working — continue. Climbing rate slowing, expect smaller gains.
- **If W5 Y similar (1.4-1.49)**: peak is local — try ensemble's x6 (0.81) instead of centroid's (0.78) as a small variation.
- **If W5 Y drops (< 1.4)**: hybrid overshot — pull back, try smaller perturbation from current best.

### F6 (5D, 25 pts after W05 query)
- Week 05 query: **(0.4248, 0.4235, 0.5083, 0.7741, 0.0543)** — RMSE-weighted ensemble of 5 interior models. Distance 0.036 from current best.
- Decision branch: #5 (STRONG consensus all 5 dims; GP-Matern only 1.15× ratio)
- F6 trajectory: W2(-0.44) → W3(-0.31) → W4(-0.30) — improvement slowing, looking like plateau
- Model fit (24 pts): all 8 beat baseline. SVR +62.6%, GP-Matern +50.0%, NN +40.5%. Ridge/SVR/NN excluded as boundary.
- Boundary-consensus watch: x5 has 2 non-Ridge models pushing < 0.02 (SVR + NN), r=-0.68. One short of the ≥3 threshold. If a 3rd model agrees next week, x5 will get clipped.
- NN gradient at best: all negative (push everything down) — corroborates ensemble's slight downward shift on x2/x4
- **If W5 Y > -0.30 (new best)**: ensemble works for the plateau — continue. Expect small gains.
- **If W5 Y similar (-0.35 to -0.30)**: peak is local — try GP-Matern's pick (0.370, 0.406, 0.513, 0.734, 0.052) as alternative direction.
- **If W5 Y drops (< -0.40)**: ensemble overshot — pull back closer to current best with smaller perturbation.

### F5 (4D, 25 pts after W05 query)
- Week 05 query: **(0.3359, 0.9073, 0.9518, 0.9378)** — RMSE-weighted ensemble of 5 interior models. Distance 0.047 from current best.
- Decision branch: #5 (STRONG consensus all 4 dims, KNN dominant only 1.17×)
- F5 has climbed every week W1-W4: 988 → 1207 → 1412 → **1979** (+47% jump in W4 — far above models' predictions)
- Model fit (24 pts): all 8 beat baseline. NN +77.4%, GP-Matern +67.5%, KNN +62.3%. Ridge/GP-Matern/NN excluded as boundary.
- Direction: x1 +0.025 (continuing to rise), x2 stable, x3 +0.004 (saturating near 0.95), x4 +0.040 (still climbing). NN gradient confirms all positive.
- **Watch**: x3 at 0.952 is close to 0.98 boundary. GP-Matern wanted 0.994 (excluded). Likely need to cap x3 ~0.96 next week.
- **Bracketing logic for outcomes**: the only meaningful per-dim moves W4→W5 are x4 (+0.040) and x1 (+0.025). x2 and x3 are essentially fixed. If Y drops, that brackets the peak between W4 and W5 picks — the midpoint is the natural next query.
- **If W5 Y > 1979 (new best)**: ensemble was conservative — keep pushing same direction. The W4 jump (+47%) suggests the function still has room.
- **If W5 Y similar (1700–1979, plateau)**: ridge of similar values — try smaller perturbation (~0.020 step) in same direction or test GP-Matern's interior pick (similar area but different spot).
- **If W5 Y drops (< 1500) — BRACKETING**: peak is between W4 and W5. Take per-dim midpoint of W4 query (0.311, 0.908, 0.948, 0.898) and W5 query (0.336, 0.907, 0.952, 0.938) → next pick ≈ **(0.324, 0.908, 0.950, 0.918)**. This works because the only big move was x4; midpoint isolates where the optimum on x4 sits.
- **If W5 Y drops a lot (< 1000)**: not just overshoot — possibly hit an unrelated local valley. Pull back closer to W4 query: (0.318, 0.908, 0.949, 0.908) with x4 only halfway between W4 and the midpoint.
- **Sanity check**: x2 has barely moved (0.876 → 0.908 over 4 weeks). If models keep agreeing on x2 ~0.91, that's likely the optimum on x2 → focus future tuning on x1, x3, x4 only.

### F4 (4D, 35 pts after W05 query)
- Week 05 query: **(0.3675, 0.4005, 0.3952, 0.4086)** — RMSE-weighted ensemble of 6 interior models. Distance 0.028 from current best (smaller than W4's 0.048).
- Decision branch: #5 (STRONG consensus on all 4 dims, GP-Matern only 1.13× ratio over SVR — not dominant)
- W4 result: ensemble took us from W1 best 0.37 to W4 0.54 (+47%). Ensemble approach validated.
- Model fit (34 pts): all 8 beat baseline. GP-Matern +74.7%, SVR +71.3%, NN +55.4%. Ridge & NN excluded (boundary).
- F4 has very narrow peak — small steps essential. W2 perturbation +0.07 → -1.39, W3 +0.06 → -0.03, W4 +0.048 → +0.54. W5 step is 0.028, even smaller.
- NN gradient at best: dx1=-15.4 (push x1 down hard), dx2=+14.7 (push x2 up hard), dx3=-6.3, dx4=-0.9. Direction informational only since NN itself was boundary-rejected.
- **If W5 Y > 0.54 (new best)**: ensemble keeps working — continue same approach. Step might shrink further next week.
- **If W5 Y ~ 0.3-0.5 (slight regression or plateau)**: peak is even narrower than thought — try midpoint between W5 query and current best (step ~0.014).
- **If W5 Y < 0 (overshoot like W2)**: pull back to GP-Matern's solo pick (the dominant model) as alternative.
- **If W5 Y << -1 (catastrophic)**: revert to pure exploitation of current best with tiny step (~0.01).

### F3 (3D, 20 pts after W05 query)
- Week 05 query: **(0.5145, 0.5469, 0.3401)** — GB dominant model (1.93× ratio over RF, after outlier removal)
- Decision branch: #2 (one model dominates and is interior)
- Outlier still present at (0.15, 0.44, 0.99) Y=-0.40 — gap of 0.27 to 2nd-worst confirms it's outlier-driven. Always train without it.
- W4 result validated GB calibration: predicted -0.0348, returned -0.0469 (off by 0.012, but didn't beat best)
- True correlations (no outlier): r(x1)=-0.004, r(x2)=+0.107, r(x3)=-0.161 — all WEAK. The signal sits in non-linear interactions that GB captures but other models miss.
- Step from current best: 0.106 (moves x1 up, x2 down, x3 unchanged at 0.34)
- RF alternative (deferred): (0.61, 0.78, 0.08) — different region, low x3 cluster near top-2
- **If W5 Y improves (> -0.0348)**: GB calibrated → continue with GB next week. Refit with 20 pts.
- **If W5 Y similar (-0.05 to -0.035)**: F3 is a flat plateau — try the RF alternative (0.61, 0.78, 0.08) for a real direction change next week.
- **If W5 Y worse (-0.10 to -0.05)**: GB overshot or the surface is sharper than expected — pull back halfway between W5 query and current best.
- **If W5 Y much worse (< -0.10)**: GB direction is wrong → switch to RF alternative or pure exploitation of current best with smaller perturbation.

### F1 (2D, 15 pts after W05 query)
- Week 05 query: **balanced Voronoi at (0.2287, 0.5658)** — Q2, the most undersampled quadrant (only 1 existing pt)
- Decision branch: #1 (no models beat baseline; combined approach failed trust check)
- Why combined approach was rejected: classifier accuracy DROPPED W4→W5 (92.3% → 71.4% best at C=10, matching majority-class). Adding the W4 positive at (0.665, 0.437) near the Q1 negative confused the boundary. Combined candidate (0.72, 0.63) sits 0.089 from a known negative; log-SVR mis-calibrated at current best by 1 order of magnitude.
- Quadrant coverage post-W5 (assuming Q2 query lands): Q1=7, Q2=2, Q3=4, Q4=2 — Q2 still tied with Q4 for least-sampled
- **F1 outliers reminder**: pt4 (0.65, 0.68) Y=-3.6e-3 and pt10 (0.42, 0.46) Y=-6.6e-3 are SIGNAL not noise — keep them in
- **If W5 Y at (0.23, 0.57) is positive (≈ 0)**: Q2 is positive territory → next week target Q3 balanced Voronoi (0.274, 0.309) or re-evaluate combined approach with Q2 data added
- **If W5 Y is large negative (< -1e-3)**: a 3rd negative observation gives log-SVR much more anchor data — reattempt combined approach next week with stricter trust checks
- **If W5 Y is LARGE positive (> 0.01) — EXIT CONDITION**: switch to exploitation around (0.23, 0.57); we found a peak region
- Long-term: continue space-filling per `f1_long_term_strategy.md` until classifier LOO ≥ 85% AND combined candidate stays > 0.15 from any negative AND log-SVR calibration improves
