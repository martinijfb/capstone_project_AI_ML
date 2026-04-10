# Suggestions for Week 04

Collected from week 03 decision markdown cells and analysis.

## Per-Function Recommendations

### F1 (2D, 13 pts)
- Week 03 query: classifier + log-SVR combined candidate at ~(0.70, 0.70)
- If Y is **positive and > 7.7e-16**: the classifier approach worked — refine near this point
- If Y is **~0 or negative**: begin **Phase 1 (space-filling)** from `suggestions/f1_long_term_strategy.md` — Voronoi largest empty circle for 3-4 weeks to build coverage, then return to classifier approach with better data
- First Voronoi target: ~(0.66, 0.44), radius 0.245

### F2 (2D, 13 pts)
- Week 03 query: RMSE-weighted average of KNN/RF/GB at ~(0.70, 0.95)
- If Y **improves** (> 0.6112): higher x2 works — continue pushing x2 (try ~0.97) while keeping x1 ~0.70
- If Y **similar** (0.4–0.6): x2=0.95 not better than 0.93 — try x1 slightly lower ~0.65 with x2=0.93
- If Y **drops** (< 0.4): overshot x2 — try smaller perturbation (0.70, 0.94)
- Note: Ridge/SVR extrapolate x1 to boundary — ignore their suggestions. x1 ~0.70 is the sweet spot.

### F3 (3D, 18 pts)
- Week 03 query: Y-weighted centroid of top 4 at ~(0.52, 0.62, 0.18)
- If Y **improves** (> -0.035): centroid is converging — refine with top 5 centroid or small perturbation near the new best
- If Y **similar** (-0.04 to -0.035): flat region — try perturbation near pt4 (0.49, 0.61, 0.34) with small random step
- If Y **worse**: centroid pulled wrong way — try pure exploitation near pt4 only
- With 18 pts, re-check if models start beating baseline (especially GB, Ridge)

### F4 (4D, 33 pts)
- Week 03 query: SVR suggestion at ~(0.40, 0.43, 0.40, 0.45) — dominant model at ~70%
- If Y **improves** (> 0.3675): SVR is calibrated — continue trusting it, re-fit with 33 pts
- If Y **similar** (0 to 0.37): very narrow peak — try midpoint between SVR suggestion and pt31
- If Y **drops** (< 0): overshot again — try even smaller perturbation from pt31 (step size 0.02)
- The peak is extremely narrow (~0.06 perturbation causes Y to drop from 0.37 to -1.39)

### F5 (4D, 23 pts)
- Week 03 query: RMSE-weighted model average at ~(0.27, 0.88, 0.92, 0.88) — continues climbing
- If Y **improves** (> 1207): still climbing — continue with same approach, re-fit models
- If Y **plateaus** (~1100-1200): near the peak — try smaller perturbation from the new best
- If Y **drops** (< 1000): overshot — try midpoint between last two bests
- Climbing trajectory: pt21 (984) → pt16 (1089) → pt22 (1207) — consistent gains

### F6 (5D, 23 pts)
- Week 03 query: SVR suggestion at ~(0.39, 0.37, 0.51, 0.85, 0.05) — dominant at 55%
- If Y **improves** (> -0.437): SVR is calibrated — continue trusting it
- If Y **similar** (-0.6 to -0.44): try centroid of top 4 as alternative
- If Y **drops** (< -0.6): SVR overshot — smaller perturbation from pt21
- Key dims: x5 (low = better, r=-0.63) and x4 (high = better, r=0.58)

### F7 (6D, 33 pts)
- Week 03 query: hybrid (centroid + model override on STRONG dims x1, x5)
- If Y **improves** (> 1.365): hybrid working — continue with same approach
- If Y **similar** (0.9–1.4): try pure SVR suggestion as alternative
- If Y **drops** (< 0.9): try perturbation near pt7 (best) with small random step
- x1 (STRONG: low ~0.04) and x5 (STRONG: ~0.37) are the most reliable dimensions

### F8 (8D, 43 pts)
- Week 03 query: hybrid (centroid + model override on STRONG dims x1, x3, x4)
- If Y **improves** (> 9.865): hybrid converging — continue same approach
- If Y **plateaus** (~9.8): near peak — try pure GP-Matern suggestion
- If Y **drops**: try centroid only (models may be overriding useful centroid values)
- Key dims: x1 (r=-0.66) and x3 (r=-0.67) — both should be LOW. All feature importances are robust.
