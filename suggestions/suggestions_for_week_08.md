# Suggestions for Week 08

Conditional next-step strategies based on Week 07 query outcomes.

---

### F1 (2D, 17 pts after W07 query)
- **W07 query**: balanced Voronoi at **(0.273645, 0.308963)** — Q3, the global Voronoi winner; pre-planned in W7 suggestions for the negative-rich region.
- Decision branch: #1 (no models beat baseline; combined approach failed two trust checks: classifier dropped to 81%, candidate 0.045 from a negative)
- W6 outcome: Y = -1.81e-57 at (0.617, 0.222) — vanishingly negative. Counted as a 5th negative point but barely informative for log-SVR.
- **WarpedRegressor (Yeo-Johnson) tested this week** — all 4 variants underperform plain models on F1. Yeo-Johnson struggles with Y magnitudes spanning 100+ orders. Don't include in F1's ensemble. Keep in pipeline for F3 + future skewed-Y functions.
- **Classifier regressed**: 87% (W6) → 81% (W7). The vanishing-negative point at (0.617, 0.222) added noise rather than signal to the SVC boundary.
- **If W7 Y is positive (≥ 0)**: Q3 has positive territory after all → revisit combined approach in W8 with all 4 quadrants now sampled. Classifier should improve.
- **If W7 Y is largely negative (< -1e-3)**: 6th meaningful negative anchors log-SVR much more — combined approach worth re-testing in W8 with stricter d_to_neg gate (≥0.20).
- **If W7 Y is LARGE positive (> 0.01) — EXIT**: switch to exploitation around (0.274, 0.309).
- Long-term: continue per `f1_long_term_strategy.md` until classifier LOO ≥ 85% AND combined candidate stays > 0.20 from any negative.

### F2 (2D, 17 pts after W07 query)
- **W07 query**: RMSE-weighted ensemble of KNN/RF/GB at **(0.703084, 0.943835)** — STRONG consensus on both dims (spread 0.039 / 0.008). Distance from current best 0.0031 (the smallest step in the project).
- Decision branch: #5 (STRONG consensus, both dims STRONG → degenerates to ensemble)
- W6 outcome: NEW BEST 0.696 (+5% over W4 best). Tiny step worked.
- **NEW finding**: BoTorch SingleTaskGP barely beats baseline (+11.9%, RMSE 0.214 vs 0.243). Better-calibrated than sklearn GPs (which still all fail at -6.7%) but well below the tree ensemble. Useful diagnostic for future GP-heavy decisions; not strong enough to displace KNN/RF/GB on F2.
- **W7 NEW**: GP-UCB (β=0.94) and qLogNEI both push x2 to boundary 1.0. Framework rejects boundary suggestions. Empirical evidence (W5 x2=0.963 → Y=0.500) shows pushing x2 too high overshoots.
- **If W7 Y improves (>0.696)**: ensemble keeps converging. Same approach W8.
- **If W7 Y in 0.55-0.69 (slight regression)**: peak is even narrower → midpoint W7 query with current best (step ~0.0015)
- **If W7 Y < 0.45 (overshoot)**: revert to W6 best (0.704, 0.947) with even smaller perturbation
- **Long-shot for W8 if W7 plateaus**: BoTorch x2-boundary push is logged; could be worth one query if ensemble approach saturates.

### F3 (3D, 22 pts after W07 query)
- **W07 query**: hybrid at **(0.557343, 0.602957, 0.175087)** — ensemble for x1/x2 (STRONG), top-4 centroid for x3 (moderate spread). Significant low-x3 push (0.340→0.175).
- Decision branch: #4 (GB dominates +55.8% but RF/Warped split on x3; centroid resolves)
- W6 outcome: Y=-0.0500 (slight regression from -0.0348). Plateau-boundary case.
- **NEW W7 finding: WarpedRegressor (Yeo-Johnson) WORKS on F3** — Warped-GB +48.2%, Warped-RF +13.7% (didn't work on F1, but does here because F3's Y range is small enough for Yeo-Johnson to fit cleanly). Now 4/14 models beat baseline (was 2/10 in W6).
- **GB switched argmax direction**: from W6's "stay near best" to (0.689, 0.603, 0.073) — the low-x3 cluster we anticipated. Models are now genuinely learning the bimodal-ish landscape.
- NN gradient at best: dx3=+0.53 (says push x3 UP) — direct conflict with our x3 down move. NN excluded (failed baseline by 112%). Informational only.
- All 4 GPs still fail (-5.3%, tied) — 4th consecutive function.
- **If W7 Y improves (> -0.0348)**: low-x3 region is real. In W8, exploit around new best with smaller step.
- **If W7 Y in -0.05 to -0.035 (plateau-ish)**: x3 push didn't fully work → try midpoint x3 (~0.26) for W8.
- **If W7 Y < -0.07 (regression)**: low-x3 was wrong → revert to current best with small perturbation, OR pull back x3 halfway.

### F4 (4D, 37 pts after W07 query)
- **W07 query**: RMSE-weighted ensemble of 7 models at **(0.368531, 0.401253, 0.431517, 0.422540)** — STRONG consensus on all 4 dims after NN outlier-filter (spread 0.113 / 0.091 / 0.074 / 0.039). Step 0.0081 from current best, smallest in F4's history.
- Decision branch: #5 (STRONG all dims, hybrid degenerates to ensemble)
- W6 outcome: NEW BEST 0.5524 (+2% over W4 0.5414). Narrow-step ensemble worked.
- Step trajectory: W4=0.048 → W5=0.028 → W6=0.028 → W7=0.008. Models converging tightly.
- **GP-Matern15 +74.7% leads** (consistent with W6). GP-RBF FAILS (-2.9%, 5th consecutive function).
- **W7 NEW**: BoTorch GP-UCB (0.390, 0.388, 0.379, 0.437) and qLogNEI (0.398, 0.390, 0.367, 0.435) both predict DOWN (0.20-0.24 vs 0.55) and move further from best. F4's narrow peak makes UCB/EI's variance-driven exploration counterproductive. Standard ensemble's tight local view is correct. Logged but rejected.
- NN gradient at best: dx1=-11.8, dx2=+19.8, dx3=+0.7, dx4=-1.6. Same as W6. NN excluded; informational.
- **If W7 Y improves (>0.552)**: peak found. In W8 try genuinely tiny step (~0.003) to see if room remains.
- **If W7 Y in 0.4-0.55 (plateau)**: F4 saturated. Consider whether to hold (current best is meaningful) or experiment with NN-gradient direction (big x1↓ x2↑) as a one-shot test.
- **If W7 Y < 0 (overshoot)**: tiny step shouldn't cause this — likely the W6 query was a lucky pick; revert with even smaller perturbation.

### F5 (4D, 27 pts after W07 query)
- **W07 query**: **TuRBO-1 q=1 candidate at (0.384873, 0.966112, 0.986585, 0.953170)** — Thompson-sampled within trust region centred on W6 best, L=0.8 init. Step 0.058 from best. **DELIBERATE FRAMEWORK DEVIATION** — used TuRBO instead of standard ensemble for the F5 climb-trajectory function.
- Decision branch: deviation from #5 — standard pipeline gave step 0.003 which would converge slowly; we documented research said "use TuRBO on F5"; this is W7 commitment to test.
- W6 outcome: NEW BEST 2669 (+15.7% over W5 2308). 6 consecutive weeks of improvement W1→W6.
- **Standard pipeline alternatives logged**:
  - Ensemble (KNN/SVR/GB/Matern05/Matern15): (0.355, 0.924, 0.969, 0.949), step 0.003 — pure exploit
  - GP-UCB (β=0.94): (0.405, 1.0, 1.0, 1.0), pred 2946 ± 171 — corner push, 1.0 violates [0,1) constraint
  - qLogNEI: (0.395, 1.0, 1.0, 1.0), pred 2962 — same corner push
- **GP-Matern05 +83.1%** still leads (W6 +79.2% → W7 +83.1%). 3rd consecutive week leading. GP-Matern25 + GP-RBF still fail.
- TuRBO state for W7 → W8: succ counter, fail counter, L all updated based on W7 result. State persisted to `data/function_5/turbo_state.json`.
- **If W7 Y > 2900 (improvement)**: TuRBO works. Continue TuRBO in W8 (state will accumulate; L stays 0.8).
- **If W7 Y in 2300-2900 (similar/slight regression)**: direction ok but x3 push to 0.987 too aggressive. In W8 try ensemble's safer x3=0.969 with TuRBO's x1/x2/x4 hints.
- **If W7 Y < 2000 (overshoot)**: TuRBO failed first time. In W8 revert to standard ensemble pipeline. TuRBO state will have fail++, L will contract toward 0.4 in W9.
- **Long-shot (Y > 5000)**: found Sterling's territory; pure exploitation in W8.
- **Peer gap reminder**: Sterling 6204 vs our 2669. Step 0.058 with all-up direction is the strongest principled move we can make without abandoning the framework entirely.

### F6 (5D, 27 pts after W07 query)
- **W07 query**: RMSE-weighted ensemble of 6 models at **(0.426310, 0.345626, 0.556770, 0.734354, 0.049005)** — STRONG consensus on all 5 dims (spread 0.157/0.064/0.091/0.069/0.040). Step 0.037 from current best, same as W6's successful step.
- Decision branch: #5 (STRONG all dims, hybrid degenerates to ensemble)
- W6 outcome: NEW BEST -0.117 (+55% over W5 -0.260). Big jump suggests we found the peak.
- **STANDARD pipeline used (not TuRBO)** — W6 was a +55% leap, this is the consolidation week. 4/6 valid models suggest identical point (extreme convergence).
- **SVR +62.9%** leads, GP-Matern05 +57.2%. GP-RBF FAILS (-4.0%, 3rd consecutive function).
- **W7 NEW logged as W8 contingency**: TuRBO suggests (0.490, 0.401, 0.679, 0.835, 0.043) — 5× larger step (0.186). GP-UCB and qLogNEI both predict Y BELOW current best (-0.20, -0.19) — they don't see another peak.
- NN gradient at best: dx1=-0.38, dx2=-0.23, dx3=-1.32, dx4=-0.62, dx5=-1.20 (push everything down). NN excluded; informational.
- **If W7 Y > -0.117 (improvement)**: ensemble keeps converging. Same step in W8.
- **If W7 Y in -0.30 to -0.12 (plateau)**: W6 was a peak; W7 didn't extend. In W8 try **TuRBO's bigger step** to test if there's another region.
- **If W7 Y < -0.40 (overshoot)**: rare given step size; pull back to W6 best with smaller perturbation.

### F7 (6D, 37 pts after W07 query)
- **W07 query**: hybrid at **(0.031697, 0.473010, 0.164637, 0.217860, 0.330911, 0.883237)** — boundary-consensus REFINED on x1, ensemble on x3/x6 (STRONG), top-4 centroid on x2/x4/x5. Half-step recovery on x3 (W6=0.143 → W7=0.165).
- Decision branch: #4 (weak global, STRONG only on x1/x3/x6, plus boundary-consensus on x1)
- W6 outcome: REGRESSION Y=1.4147 (-12% from best 1.608). x3=0.143 was too aggressive a pull-down.
- **GP-Matern05 +60.2%** still leads (4th consecutive function). GP-RBF + GP-Matern25 fail.
- **THE x3 DISAGREEMENT**: standard ensemble pushes x3 LOW (0.165), TuRBO/GP-UCB/qLogNEI all push x3 HIGH (0.75-0.96). corr(x3,Y)=-0.075 (lowest of all dims), so direction is genuinely ambiguous in the data.
- **W8 contingency** if W7 also fails: deliberately test BoTorch's x3=0.75 hypothesis as exploration query. This would be a single deliberate exploration with current best as fallback.
- NN gradient at best: dx3=-1.38 (matches standard ensemble's LOW direction). dx1=-1.88 (matches boundary-consensus). dx5=-1.24 (centroid stable).
- **If W7 Y > 1.608 (recovery)**: pull-back direction was right → continue tighter steps in W8
- **If W7 Y in 1.4-1.6 (still around W6)**: x3 LOW direction may be wrong → **W8 test BoTorch's x3≈0.75 hypothesis** as deliberate exploration
- **If W7 Y < 1.3**: pull-back also overshot → revert to current best, perturb only x6 (which has consistent positive corr)

### F8 (8D, 47 pts after W07 query)
- **W07 query**: hybrid at **(0.162411, 0.233951, 0.063190, 0.248712, 0.707848, 0.740771, 0.216942, 0.589651)** — ensemble for x1-x5/x7/x8 (STRONG, 7 dims), top-4 centroid for x6 (moderate spread). Continues W6's successful x5 push direction (+0.036 from 0.672 to 0.708).
- Decision branch: #5 (STRONG on 7/8 dims, hybrid for x6 moderate)
- W6 outcome: NEW BEST 9.911 (+0.04 from 9.868). Aggressive x5 push (0.503→0.672) paid off.
- **All 10 models beat baseline** (2nd consecutive week). SVR +80.7%, **GP-Matern25 +76.2%, Matern15 +75.4%, RBF +73.6%** — F8 still favors smoother kernels (opposite to F4-F7 pattern).
- **W7 NEW logged as W8 contingency**: TuRBO, GP-UCB, qLogNEI ALL push x5 to boundary 1.0 (matching NN gradient dx5=+0.97). If W7 plateaus, W8 should test x5=1.0 hypothesis.
- NN gradient at best: dx1=+0.48, dx2=-0.21, dx3=+0.66, dx4=-0.17, dx5=+0.97 (strongest), dx6=+0.45 (says x6 UP — centroid keeps stable), dx7=-0.24, dx8=-0.29
- **If W7 Y > 9.95**: x5 push trajectory works. Continue same approach W8. In W9, consider BoTorch x5=1.0 hypothesis.
- **If W7 Y in 9.85-9.95 (plateau)**: ensemble converging. In W8 try TuRBO/GP-UCB's x5=1.0 candidate as exploration.
- **If W7 Y < 9.7 (regression)**: pull back to W6 best with smaller perturbation.
