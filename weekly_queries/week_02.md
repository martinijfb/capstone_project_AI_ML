# Week 02 — Queries Submitted

**Date:** 2026-04-02

## Formatted Queries (portal format)

```
Function 1: 0.780000-0.780000
Function 2: 0.750000-0.920000
Function 3: 0.481100-0.532600-0.051800
Function 4: 0.454200-0.474300-0.411300-0.397000
Function 5: 0.239400-0.854200-0.899800-0.874600
Function 6: 0.283800-0.258700-0.507700-0.968700-0.043900
Function 7: 0.064000-0.482000-0.281000-0.123000-0.488000-0.676000
Function 8: 0.190945-0.227663-0.078090-0.187898-0.510101-0.741447-0.223805-0.717488
```

## Methods Used

| Function | Method | Week 01 Result | Week 02 Strategy |
|----------|--------|----------------|------------------|
| F1 | Directional perturbation up-right from pt2 | Y=-0.007 (non-zero signal, but negative) | Step away from negative pts toward the source |
| F2 | Exploit near pt9, x1 pushed higher | Y=0.281 (confirmed bimodality, valley) | Perturb near best peak, follow x1 correlation |
| F3 | GB depth=2 suggestion (only model to beat baseline) | Y=-0.112 (centroid failed) | Outlier pt6 was inflating x3 correlation. GB found on clean data |
| F4 | GP Matern suggestion | Y=+0.367 (massive improvement!) | GP has best function understanding, moderate step from new best |
| F5 | Opposite-direction perturbation from failed w1 | Y=984.4 (dropped from 1088, steep gradient) | Mirror the failed direction — if w1 went downhill, try uphill |
| F6 | SVR (C=100, gamma=auto) — GridSearchCV best | Y=-0.437 (improved from -0.714) | GridSearchCV found C=100 beats C=1. Push x4 high, x5 low |
| F7 | KNN (K=3, distance-weighted) — GridSearchCV best | Y=1.125 (x6 push to 0.90 failed) | Return near pt6. Models agree x6 should be ~0.73 not 0.90 |
| F8 | Hybrid GP (x1,x3,x7) + centroid (rest) | Y=9.804 (improved from 9.599) | Same hybrid approach, now trusting GP on x7 too |

## Key Learnings from Week 01

- **F3 outlier pt6** was inflating x3 correlation from r=-0.56 to r=-0.11. Always exclude pt6.
- **F5 gradient is extremely steep** — 0.027 step caused 104-point Y drop.
- **F7 x6 push was wrong** — models now agree x6 should stay near 0.73.
- **GridSearchCV** (sklearn) replaced manual model configs — finds better hyperparameters systematically.
- **Never hardcode values** in print statements — everything computed from data.
