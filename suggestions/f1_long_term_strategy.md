# F1 — Long-Term Strategy

## Problem

After 12 queries, ALL Y values are essentially zero (best is 7.7e-16). No model beats baseline on raw Y. The function likely has a very localised peak that we haven't found. We are data-starved and the current sample is heavily biased toward the high x1/high x2 quadrant (6 of 12 points).

## Phase 1: Systematic Space-Filling (Weeks 04–07, if week 03 classifier fails)

Spend 3–4 weeks using Voronoi largest empty circle to fill coverage gaps. The goal is NOT to find the peak directly, but to:

1. Build a well-distributed dataset across the full [0,1]^2 space
2. Discover which regions are positive vs negative (feed the classifier)
3. Ensure every quadrant has at least 3 points

### Priority regions (current coverage):
- **Low x1, high x2 (Q2):** 1 point only (pt1 at 0.32, 0.76) — needs 2+ more
- **High x1, low x2 (Q4):** 1 point only (pt4 at 0.84, 0.26) — needs 2+ more
- **Low x1, low x2 (Q3):** 4 points — adequate but clustered
- **High x1, high x2 (Q1):** 6 points — oversampled, skip

### Week-by-week plan:
- **Week 04:** Voronoi largest empty circle (currently ~(0.66, 0.44), radius 0.245)
- **Week 05:** Next largest empty circle (recompute after week 04 point added)
- **Week 06:** Target whichever quadrant still has fewest points
- **Week 07:** Final space-filling point if needed

After each week, recompute the sign classifier to see if the boundary becomes clearer.

## Phase 2: Return to Classifier + Log-SVR Combined (Week 08+)

With ~17 well-distributed points:
1. Re-train the SVM classifier on sign(Y) — should have much better accuracy with balanced coverage
2. Re-fit log-space SVR — better signal mapping with more data
3. Use the combined score (P(positive) * signal strength) to identify the peak candidate
4. The decision boundary will be much more reliable with 3-4x more data in undersampled regions

## Why This Works

- In 2D, 16-17 points with good coverage gives roughly 4x4 grid density — enough for a classifier to draw a meaningful boundary
- Each Voronoi point is guaranteed to be maximally distant from all existing samples — no wasted queries
- The classifier only needs to know positive vs negative, not the actual Y value — even ~0 values with known sign are useful training data
- By the time we return to the combined model, we'll have seen enough of the space to know where the positive region actually is

## Exit Conditions

- **If any Voronoi point returns a large positive Y (> 0.01):** immediately switch to exploitation around that point — we found the peak region
- **If the classifier boundary becomes very clear (>85% LOO accuracy) before week 08:** can return to combined approach early
- **If all ~17 points remain ~0:** the function may genuinely not have a significant peak, and the best strategy is to accept ~0 as our result
