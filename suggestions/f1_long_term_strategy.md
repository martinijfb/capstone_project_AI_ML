# F1 — Long-Term Strategy

## Problem

F1 is a 2D function where Y values span ~120 orders of magnitude: two large negatives (~-1e-3) and everything else numerically ~0 (positives range 1e-7 down to 1e-124). No raw regressor beats `Y.std()` baseline — SVR, GP, RF, GB, Ridge, KNN, NN all fail. The function likely has a very localized peak we haven't found.

**On outliers — F1 vs F3:** F1 has two "outliers" with |Y| ~10¹⁸× larger than the rest (pt at (0.42, 0.46) Y=-6.63e-3 and pt at (0.65, 0.68) Y=-3.61e-3). **These are SIGNAL, not noise — do NOT remove them.** They are the only points with non-trivial |Y| and define the entire classifier boundary and log-SVR magnitude. This is the opposite of F3 where the outlier distorted x3 correlations. Always verify whether an outlier is signal or distortion before removing — in F1 they stay.

## Primary Approach: Classifier + Log-SVR Combined (ALWAYS RUN)

Every week, **first try this approach** before anything else:

1. **Classifier** on `sign(Y >= 0)`:
   - Grid-search SVC(kernel='rbf') across C ∈ {1, 10, 100} via LOO accuracy
   - Also compare Logistic Regression and KNN as sanity check
   - Target ≥85% LOO accuracy before fully trusting the classifier
2. **Log-SVR** on `log10(|Y| + 1e-200)`:
   - GridSearchCV over C ∈ {0.1, 1, 10, 100} and gamma ∈ {'scale', 'auto'}
   - This captures magnitude structure — where is the function most active?
3. **Combined score** across dense grid: `P(positive) × normalized log|Y|`
   - Best candidate = `argmax(combined | P(positive) > 0.5)`

### When to trust the combined candidate

| Condition | Action |
|---|---|
| Classifier LOO ≥ 85% AND candidate is far from known negatives (> 2 × avg pairwise distance) | **Use combined candidate** |
| Classifier LOO ≥ 85% but candidate sits on the sign boundary | Mistrust — log-SVR extrapolates magnitude from negatives → fall back to balanced Voronoi |
| Classifier LOO < 85% | Not enough sign-structure learnable → fall back to balanced Voronoi |
| Combined predicted log|Y| is consistent with observed positive magnitudes | Trust |
| Combined predicted log|Y| is much higher than any observed positive | Mistrust (extrapolation from negatives) → fall back |

### Sanity checks to print every week

- Log-SVR prediction at the **current best point** vs. actual log|Y|. If off by >1 order of magnitude, log-SVR is miscalibrated.
- Top-K grid-cells by raw log|Y| prediction: are they near known negatives (bad) or interior to positive region (good)?
- Distance from combined candidate to nearest known negative.

## Fallback: Balanced Voronoi Space-Filling

When the combined approach isn't trustworthy (see conditions above), space-fill to build better training data for next week.

**Important**: use **balanced Voronoi** — `max(min(dist_to_data, dist_to_boundary))` — NOT raw Voronoi. Raw Voronoi picks corners, which are degenerate:
- BBO functions often behave atypically at edges
- A corner only probes "one side" of the decision boundary
- Low information value for the classifier

The balanced metric penalizes corners equally with cluster-proximity, producing a genuinely interior point.

### Quadrant coverage priority

| Quadrant | Priority |
|---|---|
| Q1 (hi x1, hi x2) | Low — oversampled (6+ pts) |
| Q2 (lo x1, hi x2) | **High** — undersampled |
| Q3 (lo x1, lo x2) | Medium |
| Q4 (hi x1, lo x2) | **High** — undersampled |

Pick the undersampled quadrant with the largest balanced-empty-circle radius.

## Exit Conditions

- **If any query returns a large positive Y (> 0.01)**: immediately switch to exploitation around that point — we found the peak region. Ignore the space-filling plan.
- **If all ~17 points remain ~0 after Phase 1**: the function may genuinely not have a significant peak — accept ~0 as our result and allocate the remaining weekly queries to other functions that have more room to improve.

## Weekly Decision Tree

```
For F1, every week:
  1. Run model grid search (Ridge/KNN/RF/SVR/GB/GP/NN) — confirm baseline fails (expected)
  2. Run classifier + log-SVR combined approach
  3. Check trust conditions for combined candidate
     ├── Trustworthy → use combined candidate
     └── Not trustworthy → balanced Voronoi in most undersampled quadrant
  4. Update this file if any assumptions change
```

## Reference: observed positives (as of W04)

All 9 positive Y values are ≤ 4e-7 — effectively zero. The log-SVR has no real positive-magnitude training signal, which is why it extrapolates magnitude from the two large negatives. Until we observe a non-trivial positive, the combined approach's log|Y| predictions in positive regions are unreliable.
