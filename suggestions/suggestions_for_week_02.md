# Suggestions for Week 02

Collected from the week 01 decision markdown cells and classmates' reflections.

## Per-Function Recommendations (from Week 01 analysis)

### F1 (2D, 11 pts after results)
- If non-zero signal: switch to log-gradient analysis or GP+UCB
- If still zero: try Latin Hypercube Sampling for systematic coverage
- Several classmates also planning grid sweep / LHS for F1

### F2 (2D, 11 pts)
- If Y at [0.754, 0.525] is high: function has a ridge → push x1 further right (0.85-0.95)
- If Y is low: bimodal → exploit near pt9 [0.70, 0.93]
- If moderate: try GP+UCB with 11 points (should have enough signal now)

### F3 (3D, 16 pts)
- If Y improves: perturb slightly, push x3 lower toward 0.10
- If not: try near pt13 [0.600, 0.725, 0.066] (lowest x3 among top performers)
- Re-evaluate whether models can now beat baseline with 16 points
- Exploration targets: [0.83, 0.50, 0.17], [0.50, 0.17, 0.17], [0.83, 0.83, 0.17]

### F4 (4D, 31 pts)
- If Y improves (better than -4.0): re-fit SVR with 31 points
- If similar: perturb around pt27 with lower x4
- If worse: fall back to centroid or ensemble
- Try tuning Ridge alpha on polynomial features (boundary fix)
- Push x4 lower (0.10-0.15) in a future query

### F5 (4D, 21 pts)
- If Y > 1088: continue perturbation in direction of improvement
- If Y ≈ 1088: larger perturbation to map peak shape
- Test whether x4 alone drives performance: [0.5, 0.5, 0.5, 0.95]
- x3 importance was inflated by pt15 — don't assume high x3 matters away from peak

### F6 (5D, 21 pts)
- If Y improves: re-fit SVR with 21 points
- If not: try centroid or push x5 even lower
- Check whether x3 matters (best point has x3=0.73 but weak correlation)

### F7 (6D, 31 pts)
- If Y improves (> 1.365): x6 push worked → push x6 higher or combine with lower x5
- If similar: try perturbing x5 lower instead
- If worse: stay closer to pt6's x6=0.73
- x1 importance was inflated by pt6 — real drivers are x5, x2, x6

### F8 (8D, 41 pts)
- If Y improves: x1/x3 push worked → re-run GP, consider trusting more dimensions
- If similar: try perturbing x7 lower (next strongest signal)
- If worse: exploit closer to pt14
- GP should be even more reliable with 41 points

## Ideas from Classmates

- **Kappa/beta decay** over time (several classmates planning this)
- **Latin Hypercube Sampling** for F1 (Athanasios, Sterling)
- **Weighted mix of top performers** — Venkata uses 60/30/10 weighting of top 3 (similar to our centroid but different weights)
- **Nick Mara** had the same boundary issue we caught — models suggesting edge points
- **Multiple acquisition functions** (Mark Jones) — compare UCB, EI, PI as a cross-check

## Techniques Becoming Available (Module 13+)
- Module 13: Logistic regression (probably not useful for regression tasks)
- Module 14: SVMs — kernel functions, soft-margin (we already use SVR)
- Module 15: Neural networks — could try as surrogate for high-dim functions
