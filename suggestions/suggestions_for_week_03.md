# Suggestions for Week 03

Collected from week 02 decision markdown cells and analysis.

## Week 01 → Week 02 Results Summary

| Function | W01 Y | W02 Strategy | Expected Outcome |
|----------|-------|--------------|------------------|
| F1 | -0.007 | Directional perturbation [0.78, 0.78] | Positive Y if source is up-right of pt2 |
| F2 | 0.281 | Exploit near pt9 [0.75, 0.92] | Beat pt9's Y=0.611 if higher x1 helps |
| F3 | -0.112 | GB suggestion [0.48, 0.53, 0.05] | Beat pt3's Y=-0.035 (pred: -0.031) |
| F4 | +0.367 | GP Matern [0.45, 0.47, 0.41, 0.40] | Beat 0.367 (GP pred: 0.68) |
| F5 | 984.4 | Opposite perturbation [0.24, 0.85, 0.90, 0.87] | Beat 1088 if opposite direction is uphill |
| F6 | -0.437 | SVR C=100 [0.28, 0.26, 0.51, 0.97, 0.04] | Beat -0.437 (SVR pred: -0.31) |
| F7 | 1.125 | KNN [0.06, 0.48, 0.28, 0.12, 0.49, 0.68] | Beat pt6's Y=1.365 |
| F8 | 9.804 | Hybrid GP+centroid | Beat 9.804 |

## Per-Function Recommendations

### F1 (2D, 12 pts)
- If Y at [0.78, 0.78] is **positive and > 7.7e-16**: source is up-right → continue stepping in that direction
- If Y is **negative**: overshot → try smaller perturbation [0.74, 0.74]
- If Y is **~0 again**: flat region → switch back to exploration (LHS targeting unexplored quadrants: low x1/high x2 or high x1/low x2)

### F2 (2D, 12 pts)
- If Y > 0.611: higher x1 helps → push x1 further (try x1≈0.80)
- If Y < 0.611 but > 0.4: x1=0.75 slightly worse → try opposite direction x1≈0.65 with x2 still high
- If Y drops significantly: peak is sharp → smaller perturbation [0.72, 0.93]

### F3 (3D, 17 pts)
- If Y improves (better than -0.035): GB is working → re-fit with 17 points, try more GB configs
- If Y is similar: flat near optimum → perturb near pt3 [0.493, 0.612, 0.340] instead
- If Y is worse: GB was wrong → fall back to perturbation near pt3
- Always exclude pt6 from analysis (outlier inflating correlations)

### F4 (4D, 32 pts)
- If Y > 0.367: GP was right → continue with GP, possibly with UCB for more exploration
- If Y ≈ 0.367: flat region → try SVR's more conservative suggestion
- If Y < 0: overshot → smaller perturbation [0.39, 0.43, 0.41, 0.40]

### F5 (4D, 22 pts)
- If Y > 1088: found the uphill direction → continue stepping further
- If Y ≈ 984 (similar to w1): peak is symmetric, both sides drop equally → try much smaller perturbation (dist=0.01)
- If Y drops further: try **GB's bold suggestion [0.432, 0.850, 0.870, 0.880]** — completely different x1, GB predicted Y=1090 with 53% LOOCV improvement
- Consider one-dimension-at-a-time perturbation to isolate which dim matters most near the peak

### F6 (5D, 22 pts)
- If Y improves: SVR C=100 working → re-fit with 22 points, follow updated suggestion
- If Y doesn't improve: try GP's more aggressive suggestion (lower x1, x2)
- Check if x4 should be pushed even higher (we sent x4=0.97)
- Monitor x5=0.044 — is there a floor?

### F7 (6D, 32 pts)
- If Y > 1.365: KNN perturbation improved on pt6 → continue in this direction
- If Y ≈ 1.125 (similar to w1): flat region → try SVR's slightly different suggestion
- If Y < 1.0: try very small perturbation from pt6 (dist < 0.05)
- Lesson: don't push a single dimension far from the best point

### F8 (8D, 42 pts)
- If Y improves again: hybrid approach consistently works → continue, consider trusting GP on x2 (r≈-0.26)
- If Y plateaus: centroid dimensions may be holding back → try full GP suggestion
- If Y drops: x7 override may have been wrong → revert to x1/x3 only

## Techniques from Module 13
- **Logistic regression** — could be used to classify "good" vs "bad" regions as a pre-filter before running surrogate models
- Consider using logistic regression to define decision boundaries for functions with clear threshold behaviour (F8 at Y≈9.0)

## Key Principles Established
- **GridSearchCV** for all model selection (never manual configs)
- **Feature importance robustness check** always (with/without best point AND outliers)
- **Never hardcode values** — all analysis computed from data
- **Always add code cells to notebook** — both internal analysis and notebook cells
- **Outlier exclusion**: F3 pt6 should always be removed
