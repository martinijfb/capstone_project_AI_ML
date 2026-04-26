# Week 05 Queries — Submitted 2026-04-22

## Formatted Queries

```
Function 1: 0.228696-0.565819
Function 2: 0.693851-0.962584
Function 3: 0.514480-0.546868-0.340101
Function 4: 0.367519-0.400462-0.395157-0.408596
Function 5: 0.335874-0.907262-0.951751-0.937792
Function 6: 0.424840-0.423462-0.508314-0.774109-0.054349
Function 7: 0.054203-0.467824-0.220489-0.215666-0.316569-0.779411
Function 8: 0.084770-0.218794-0.051901-0.179535-0.502886-0.740713-0.157593-0.726102
```

## Methods

| F | Method | W4 Result | Predicted / Note |
|---|--------|-----------|-------------------|
| F1 | Balanced Voronoi in Q2 (most undersampled quadrant, only 1 existing pt) | ~0 (no peak in Q4) | Pure space-filling — combined approach failed trust check (classifier dropped to 71%, candidate 0.089 from negative) |
| F2 | RMSE-weighted ensemble of 5 interior models (KNN/RF/GB/GP-Matern/GP-RBF) — STRONG consensus on both dims | 0.0852 (low-x2 ridge bet failed) | Ensemble pred ~0.65; tiny step (0.012) from current best 0.6658 |
| F3 | GB dominant model (1.93× ratio, +56% vs no-outlier baseline) | -0.0469 (close to GB pred -0.0348) | GB predicts -0.0353 (matches current best -0.0348) |
| F4 | RMSE-weighted ensemble of 6 interior models — STRONG consensus on all 4 dims | 0.5414 (NEW BEST) | All 8 models beat baseline; GP-Matern +75% top. Step 0.028 from best (smaller than W4's 0.048). |
| F5 | RMSE-weighted ensemble of 5 interior models — STRONG consensus on all 4 dims | 1979.02 (NEW BEST) | NN +77% best, GP-Matern +68%. Step 0.047 from best. **Watch x3 boundary** (currently 0.952, 0.028 from cap) |
| F6 | RMSE-weighted ensemble of 5 interior models — STRONG consensus on all 5 dims | -0.3035 (NEW BEST, marginal) | SVR +63%, GP-Matern +50%. Boundary-consensus on x5 at 2/3 threshold (one short) |
| F7 | Hybrid: ensemble on STRONG dims (x1-x5) + Y-weighted top-4 centroid on weak dim (x6) | 1.4925 (NEW BEST) | Climbing slow: 1.125 → 1.365 → 1.461 → 1.493. GP-Matern +50% top. Same hybrid pattern from W3-W4. |
| F8 | Hybrid: ensemble on STRONG dims (x1, x3, x7) + centroid on weak/moderate (x2, x4, x5, x6, x8). NO boundary-consensus this week. | 9.8518 (slight regression) | x3 boundary-consensus dropped from 4 models (W4) to 1 (W5) — rule self-corrected after W4 push didn't help. GP-Matern +84% top. |

## New Pattern Observed This Week

**The boundary-consensus rule self-corrects.** F8 W4 had 4 non-Ridge models pushing x3 < 0.02; we clipped x3 to 0.023 (top-5 min). The query returned Y=9.8518, slight regression from the W2 best of 9.8651. This week, only 1 model still pushes x3 low — the new data point at x3=0.023 with non-best Y reduced models' confidence in pushing further low. The rule "knows" the boundary push didn't help and doesn't re-apply.

## Notable W4→W5 Model Shifts

- **F2 NN dropped from beating baseline to failing it** (-0.4%) — the W4 outlier point at (0.90, 0.12, Y=0.085) added a far-from-cluster low-Y point that broke MLP smoothness assumption
- **F7 NN improvement nearly doubled** (+13% W4 → +27% W5) — new W4 cluster point validated the local fit
- **F4 NN suggestion now boundary-rejected** (x1=0.009) — model overfitting toward a corner
- **F8 KNN/GP-RBF added to boundary-rejected list** (W4 had only Ridge/SVR/GB/GP-Matern/NN bouncing; W5 also KNN x3=0.010 and GP-RBF x4=0.003)

## Neural Network Surrogates (Week 05 retrain)

Pre-trained MLPs in `models/week_05/`:
- F1: dropout/H32 — ✗ fails baseline (-52%)
- F2: dropout/H32 — ✗ fails (-0.4%, slight)
- F3: ensemble/H32 — ✗ fails (-5.5%, trained WITH outlier)
- F4: plain/H32 — ✓ beats +55%
- F5: plain/H32 — ✓ beats +77%
- F6: ensemble/H32 — ✓ beats +41%
- F7: ensemble/H32 — ✓ beats +27%
- F8: plain/H32 — ✓ beats +63%

5/8 functions have NN surrogates that beat baseline. Width converged to H=32 across all functions.

## Notes

- Date: 2026-04-22
- Total cells: 38 in `notebooks/week_05.ipynb`
- All queries computed from data + models, no hardcoded values
- Plots saved to `plots/week_05/` (per-dim, parallel coords, F1 combined + Voronoi, F2 2D scatter, F3 outlier-aware)
- Running best Y per function: F1=0.0 (still), F2=0.67, F3=-0.035, F4=0.54, F5=1979, F6=-0.30, F7=1.49, F8=9.87
- W4 produced 4 new bests (F4, F5, F6, F7); 4 regressions (F1, F2, F3, F8). F2 was the painful one — risk gambit at (0.90, 0.12) returned 0.085 vs deferred ensemble's expected ~0.67.
- W5 strategy: return to ensemble where it worked (F2, F4, F5, F6), continue successful patterns (F3 GB, F7/F8 hybrid), test if balanced Voronoi in Q2 helps F1 classifier
