# Week 01 — Queries Submitted

**Date:** 2026-03-26

## Formatted Queries (portal format)

```
Function 1: 0.421062-0.463562
Function 2: 0.753982-0.525267
Function 3: 0.507299-0.624851-0.228441
Function 4: 0.384555-0.428956-0.409751-0.392875
Function 5: 0.209005-0.838745-0.859155-0.882438
Function 6: 0.516000-0.351285-0.511928-0.691626-0.159640
Function 7: 0.129100-0.462600-0.267000-0.237000-0.403000-0.900000
Function 8: 0.142000-0.231000-0.110000-0.208000-0.507000-0.725000-0.408000-0.710000
```

## Methods Used

| Function | Method | Predicted Y | Current Best Y |
|----------|--------|-------------|----------------|
| F1 | Voronoi largest empty circle (space-filling) | N/A (no signal) | 0.0000 |
| F2 | Y-weighted centroid top 4 (bimodality probe) | N/A (centroid) | 0.6112 |
| F3 | Y-weighted centroid top 4 (models failed baseline) | N/A (centroid) | -0.0348 |
| F4 | SVR (RBF, C=10) — 63% LOOCV improvement | -1.3526 | -4.0255 |
| F5 | GP (Matern) — perturbation near unimodal peak | 1073.54 | 1088.86 |
| F6 | SVR (RBF, C=1) — 30% LOOCV improvement | -0.6520 | -0.7143 |
| F7 | Centroid + x6 push to 0.90 (model consensus) | N/A (hybrid) | 1.3650 |
| F8 | Hybrid GP (x1,x3) + centroid (x4-x8) | N/A (hybrid) | 9.5985 |

## Decision Summary

- **F1**: No signal in data (all Y ≈ 0). Space-filling via constrained Voronoi largest empty circle.
- **F2**: Real signal but sparse (10 pts, 2D). Y-weighted centroid probes bimodality between two peaks.
- **F3**: 15 pts in 3D. 7 sklearn models tested — none beat baseline. Centroid anchored to data.
- **F4**: 30 pts in 4D. SVR dominated (63% improvement). Polynomial features confirmed quadratic structure.
- **F5**: 20 pts in 4D, unimodal. GP perturbation near peak. Feature robustness showed x3 importance was inflated by outlier.
- **F6**: 20 pts in 5D. SVR best at 30%. Strong x5 (negative) and x4 (positive) signals.
- **F7**: 30 pts in 6D. Grid search of 17 configs — only 6 beat baseline by tiny margins. Used centroid with x6 pushed higher (only dimension with strong model consensus).
- **F8**: 40 pts in 8D. GP dominated at 78.5% improvement. Trusted GP on x1/x3 (strong corr + robust importance + consensus), centroid on rest.
