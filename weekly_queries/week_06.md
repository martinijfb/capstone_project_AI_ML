# Week 06 Queries — Submitted 2026-05-03

## Formatted Queries

```
Function 1: 0.617191-0.222274
Function 2: 0.703636-0.946935
Function 3: 0.418662-0.591738-0.333919
Function 4: 0.366878-0.404857-0.433485-0.415743
Function 5: 0.354155-0.921140-0.968749-0.947704
Function 6: 0.420022-0.376592-0.537773-0.739730-0.048710
Function 7: 0.031697-0.474149-0.142789-0.217730-0.335014-0.787502
Function 8: 0.155378-0.200562-0.075482-0.215712-0.672482-0.740636-0.179584-0.616336
```

## Methods

| F | Method | W5 Result | Predicted / Note |
|---|--------|-----------|-------------------|
| F1 | Balanced Voronoi in Q4 (least-sampled positive-only quadrant) | 6.24e-42 (~zero, in Q2) | Combined classifier (LOO 87% — first time crossing 85%) + log-SVR ran but failed `d_to_neg ≥ 0.15` gate (candidate 0.049 from a negative). Matern 0.5 on rank(Y) and combined-score independently picked the same Q4 region as Voronoi. |
| F2 | RMSE-weighted ensemble of 3 interior models (KNN/RF/GB) — STRONG consensus both dims | 0.4995 (regression from 0.6658) | Step 0.007 from current best — even smaller than W5's 0.012 that lost 0.166 in Y. All 4 GPs FAIL baseline (-7.1% tied). |
| F3 | RMSE-weighted ensemble of RF + GB (GB dominant) — STRONG consensus all 3 dims | -0.0470 (no improvement, plateau) | GB +60.8% dominant, RF +21.1%. NN trained on full data fails the cleaner no-outlier baseline. Step 0.077. |
| F4 | RMSE-weighted ensemble of 7 interior models — STRONG consensus all 4 dims | 0.4055 (regression from 0.5414) | **Matern 1.5 leads +73%** (peer-derived, dethrones Matern 2.5). GP-RBF FAILS for first time. Step 0.028 (same as W5). |
| F5 | Hybrid — ensemble for x1/x2/x4, REFINED boundary-consensus for x3 | 2307.54 (NEW BEST, +17%) | **Matern 0.5 +79.2% leads** (2nd function). REFINED rule applied: x3 = max(top-5 max 0.952, ensemble 0.969) = 0.969. Step 0.030. |
| F6 | RMSE-weighted ensemble of 7 interior models — STRONG consensus all 5 dims | -0.2598 (NEW BEST, +14%) | SVR +60.1% leads, Matern 0.5 +54.5% (3rd function). GP-RBF FAILS again. Step 0.066. |
| F7 | Hybrid — boundary-consensus REFINED on x1, ensemble on x3 (STRONG), top-4 centroid on x2/x4/x5/x6 | 1.6078 (NEW BEST, +7.7%) | **Matern 0.5 +57.8% leads** (4th function). x1 boundary-consensus → REFINED clip 0.0317. **Big x3 push: 0.220 → 0.143** (NN gradient agrees). Step 0.084. |
| F8 | Hybrid — ensemble on x1/x2/x3/x4/x5/x8 (STRONG), top-4 centroid on x6/x7 | 9.8684 (NEW BEST, marginal +0.03%) | **All 10 models beat baseline** (first time). F8 reverses pattern: smoother kernels win (Matern 2.5 +75%, RBF +72%). **Big x5 push: 0.503 → 0.672** (NN gradient +0.97 strongest). Step 0.27 (most aggressive of project). |

## New Patterns Observed This Week

**1. Peer-derived Matern variants validated.** Added Matern ν=0.5 and ν=1.5 to the GP family this week (after Athanasios's W6 reflection). Result: in 4 of 8 functions (F4, F5, F6, F7), one of the new variants becomes the top GP model, dethroning Matern 2.5. F8 prefers the smoother kernels (Matern 2.5/RBF) — the only function where the original choice still wins. Worth keeping permanently.

**2. GP-RBF should be removed from future grids.** Failed baseline on F4, F5, F6, F7 (all 4 consecutive functions in this dimensionality range). Only beat baseline on F8 (8D, 45 pts where smoothness genuinely helps). 4/5 fail rate.

**3. Boundary-consensus rule REFINED.** Old rule clipped to top-5 observed extremum, which on F5's x3 would have downgraded a valid interior step. New rule: clip to `max(top-5 max, interior ensemble)` for high-edge (and `min` for low-edge). Preserves the safety against extrapolation while allowing the modest interior step that valid models support. Applied to F5 (x3) and F7 (x1) this week.

**4. Cross-architecture infrastructure fix.** Switched notebook execution from anaconda x86_64 (running through Rosetta) to the existing uv-managed native arm64 Python. Eliminates 600+ Intel MKL warning lines, removes the multiprocessing kernel-died issue, and makes numpy 3.5× faster on this M4. No package changes — pyproject.toml was already correct.

## Notable W5→W6 Model Shifts

- **F1 classifier improved to 87%** (from 71% in W5) — first time crossing the 85% trust threshold. Combined approach still rejected because candidate sat 0.049 from a negative.
- **F2 NN recovered** from -0.4% (W5) to +8.9% (W6) — the W4 outlier impact fading.
- **F3 NN trained on full data** still beats full-data baseline (+6.9%), but fails cleaner no-outlier baseline (-112%).
- **F8 saw all 10 models beat baseline** — most data (45 pts) + smoothest landscape gives every model family enough signal.

## Neural Network Surrogates (Week 06 retrain)

Pre-trained MLPs in `models/week_06/`:
- F1: ensemble/H32 — ✗ fails (-27.6%, but improving)
- F2: dropout/H16 — ✓ recovered (+8.9%)
- F3: ensemble/H32 — ✓ marginal on full baseline (+6.9%); fails on cleaner no-outlier baseline
- F4: ensemble/H16 — ✓ +50.4%
- F5: plain/H32 — ✓ +77.0%
- F6: ensemble/H16 — ✓ +43.2%
- F7: ensemble/H32 — ✓ +39.7%
- F8: plain/H32 — ✓ +64.6%

7/8 functions have NN surrogates that beat baseline (up from 5/8 in W5).

## Notes

- Date: 2026-05-03
- Total cells: 70 in `notebooks/week_06.ipynb`
- All queries computed from data + models, no hardcoded values
- Plots saved to `plots/week_06/` (per-dim, parallel coords, F1 combined + Voronoi, F2 2D scatter)
- Running best Y per function: F1=3.65e-7, F2=0.67, F3=-0.035, F4=0.54, F5=2308, F6=-0.26, F7=1.61, F8=9.87
- W5 produced 4 new bests (F5, F6, F7, F8) and 4 regressions (F1, F2, F3, F4). Climbing functions continue to climb; narrow-peak functions (F2, F4) keep over-stepping.
- W6 strategy: tighter steps on F2 (-half W5 step) and F4 (same W5 step but new direction); plateau-break attempts on F7 (big x3 move) and F8 (big x5 move) where the framework's STRONG consensus + NN gradient agree.
- Per-function decision histories in each notebook's Cell G markdown.
