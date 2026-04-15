# Week 04 Queries — Submitted 2026-04-15

## Formatted Queries

```
Function 1: 0.665351-0.437391
Function 2: 0.895800-0.123969
Function 3: 0.367928-0.456520-0.337477
Function 4: 0.369558-0.403873-0.410522-0.431202
Function 5: 0.311287-0.908307-0.947911-0.898294
Function 6: 0.423671-0.437044-0.501334-0.805860-0.046742
Function 7: 0.064412-0.481541-0.243221-0.229854-0.341439-0.812127
Function 8: 0.086114-0.216397-0.022929-0.132947-0.500534-0.739359-0.150125-0.728137
```

## Methods

| F | Method | Week 03 Result | Predicted Y |
|---|--------|----------------|-------------|
| F1 | Balanced Voronoi in Q4 — min(dist to data, dist to boundary) penalizes corners | 3.65e-7 (tiny positive, classifier+log-SVR approach) | N/A (no model beats baseline; pure space-filling for classifier training) |
| F2 | Low-x2 ridge exploration — derived from pt#3 anomaly (0.67, 0.12, Y=0.54) pushed to high x1 | 0.6658 (NEW BEST, RMSE-weighted ensemble) | SVR predicts ~0.72 at interior-safe neighborhood of the boundary direction |
| F3 | GB dominant (+55% vs baseline) after OUTLIER removal (pt7 at x=(0.15,0.44,0.99) Y=-0.40 was inflating x3 correlation) | -0.1096 (centroid failed, landed in valley between bimodal peaks) | GB predicts -0.035 (matches current best) |
| F4 | RMSE-weighted ensemble of 6 interior models (NN excluded as spatial outlier at boundary-adjacent point) — STRONG consensus all 4 dims | -0.0296 (partial recovery from W2's -1.39) | Ensemble ~+0.5 at ~0.048 distance from W1's best 0.37 |
| F5 | RMSE-weighted ensemble of 6 interior models (all 8 beat baseline, NN boundary-excluded) — STRONG consensus all 4 dims | 1412.13 (NEW BEST, continues climbing) | Models predict +1555 to +1693 |
| F6 | RMSE-weighted ensemble of 5 interior models (SVR + NN boundary-excluded) — STRONG on x2/x4/x5, moderate on x1/x3 | -0.3057 (NEW BEST, SVR-driven in W3) | GP-Matern -0.28, ensemble range -0.21 to -0.39 |
| F7 | Hybrid: RMSE-weighted ensemble on STRONG-consensus dims (x1/x3/x4/x5/x6) + Y-weighted top-4 centroid on weak dim (x2) | 1.4609 (NEW BEST, hybrid method working) | Distance 0.063 from best, matches successful W3 step pattern |
| F8 | Hybrid + **boundary-consensus clip** on x3: 4 non-Ridge models push x3 < 0.02, r(x3)=-0.69, pt#4 has x3=0.023 Y=9.60 → clip x3 to 0.023 (observed top-5 min, not extrapolated 0) | 9.8209 (slight regression from W2's 9.87) | RF+SVR ensemble + boundary-informed x3 |

## New Rules Developed This Week

1. **Balanced Voronoi** replaces raw Voronoi everywhere — penalizes corner-picking equally with cluster-adjacency via `min(d_data, d_boundary)`. Raw Voronoi always picks corners which are degenerate for BBO (boundary effects, one-sided information).

2. **Outlier-suggestion filter** — when models converge tightly, one model whose suggestion is far from the cluster centroid (> `mean + 2σ` distances) is excluded from the ensemble. Applied to F4 (NN spatial outlier).

3. **Boundary-consensus rule (F8)** — when ≥3 **non-Ridge** models push the same dim to the same edge AND correlation sign matches AND observed top-5 data has good Y near that edge, clip to the observed top-5 min/max as a "confirmed-good boundary value". Avoids extrapolation to extremes while respecting multi-model agreement.

4. **F3 outlier check** — always analyse F3 both with AND without the single outlier pt7 at (0.15, 0.44, 0.99) Y=-0.40. Without it, GB suddenly beats baseline by +55%. The W3 failure to beat baseline was caused by outlier-inflated x3 signal.

5. **F1 outliers are SIGNAL, not noise** — pt4 and pt10 with |Y| ~10¹⁸× larger than rest are the only non-trivial magnitudes. They define the classifier boundary and log-SVR's magnitude anchor. Do NOT remove them (opposite rule from F3).

## Neural Network Surrogates (Week 04 addition)

Pre-trained MLPs in `models/week_04/`:
- F1: dropout/H16 — ✗ fails baseline (-46%)
- F2: ensemble/H32 — ✓ beats +10%
- F3: ensemble/H32 — ✗ fails (-8%, trained WITH outlier)
- F4: ensemble/H16 — ✓ beats +49%
- F5: plain/H32 — ✓ beats +75%
- F6: ensemble/H16 — ✓ beats +37%
- F7: dropout/H32 — ✓ beats +13%
- F8: dropout/H16 — ✓ beats +62%

6/8 functions have NN surrogates that beat baseline. NN suggestions included in convergence analysis; NN excluded if boundary or spatial outlier.

## Notes

- Date: 2026-04-15
- Total cells: 42 in `notebooks/week_04.ipynb`
- All queries computed from data + models, no hardcoded values
- Plots saved to `plots/week_04/` (per-dim exploration, 2D scatters, parallel coords, F1 Voronoi + combined, F8 parallel)
- Running best Y per function: F1=0.0 (tiny), F2=0.67, F3=-0.035, F4=+0.37 (W1), F5=1412, F6=-0.31, F7=1.46, F8=9.87
