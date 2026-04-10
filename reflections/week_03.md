# Week 03 — Reflection

## Full Reflection

### Q1: How has your query strategy changed from earlier rounds?

The biggest change this week was moving from a one-size-fits-all model pipeline to function-specific strategies based on what the models can actually learn. In weeks 01–02, I ran the same GridSearchCV pipeline on every function and used whichever model won. This week, I developed three distinct approaches depending on model reliability:

For functions where models work well (F2, F4, F5, F6 — all with 30–70% LOOCV improvement), I trust the dominant model's suggestion directly. SVR emerged as the clear winner for F4 (70%) and F6 (55%), while F2 and F5 benefit from RMSE-weighted ensembles of multiple non-linear models.

For functions where models disagree on some dimensions (F7, F8), I built a hybrid: centroid of top performers for dimensions with weak model consensus, overridden by the model average only on dimensions where all models agree (spread < 0.2). This selective trust prevents models from dragging the query into bad territory on dimensions they don't understand.

For F1, where no model beats baseline on raw Y, I developed a novel approach: splitting regression into classification (SVM predicting sign of Y) plus log-space SVR (predicting signal strength). Neither sub-problem alone is useful, but combining P(positive) with normalised signal strength triangulates a candidate that's both confidently positive and in a high-signal region.

I also systematically exclude Ridge and SVR from ensemble averages when they extrapolate to boundary corners — a lesson from week 02 where a GP overshot badly on F4.

### Q2: Exploration vs exploitation balance?

This week was predominantly exploitation, but the balance varied by function.

**Heavy exploitation** (F2, F4, F5, F8): These functions have strong model signal (40–80% LOOCV improvement) and clear climbing trajectories. F5 has improved three consecutive weeks (984 → 1089 → 1207), so continuing the model-guided direction is well-justified. F8 also improved incrementally (9.80 → 9.87) using a GP+centroid hybrid.

**Informed exploration** (F1): All 12 points return essentially zero, so pure exploitation is meaningless. But rather than blind exploration (Voronoi), I used the classifier+log-SVR combined approach to explore the region most likely to contain the peak — the positive side of the sign boundary where signal is strongest. This is exploration guided by all available information.

**Centroid-based** (F3): With 0/6 models beating baseline at 17 points in 3D, I fall back to the Y-weighted centroid — a data-driven compromise that doesn't require trusting any model.

**Hybrid** (F7): Models agree on x1 (low) and x5 (~0.37) but disagree wildly on x2 and x4. I exploit the consensus dimensions while letting the centroid handle the uncertain ones.

The trade-off I've learned: exploitation only makes sense where models have earned trust through cross-validation. Exploiting a 5% improvement model is worse than exploring — but exploiting a 70% model is almost certainly right.

### Q3: How would SVMs change your approach?

I actually implemented this for F1 this week. Since all Y values are near zero (best is 7.7e-16), direct regression is hopeless. But an SVM classifier predicting sign(Y) achieves 75% LOO accuracy — it can distinguish which regions produce positive vs negative outputs. Combined with a log-space SVR for signal magnitude, this gives a principled search target.

A soft-margin SVM would be valuable for F5 and F8, where I could set a threshold (e.g., Y > 1000 for F5) and classify the input space into "promising" vs "unpromising" regions. The soft margin would handle the noise inherent in functions where nearby points can have very different Y values (F4's peak is so narrow that a 0.06 perturbation drops Y from 0.37 to -1.39).

A kernel SVM with RBF would help for F4 specifically — the function has confirmed quadratic structure (polynomial features gave 85% improvement in week 01), and an RBF kernel would naturally capture this non-linear boundary without manual feature engineering.

### Q4: Model limitations as data grows?

The most significant limitation is **models extrapolating to boundaries**. Ridge and SVR consistently suggest corners of the input space (e.g., x1=0.98 for F2) because they follow linear correlations to their extreme. I now systematically filter these models from ensemble averages — an expensive lesson from F4 week 02 where the GP overshot.

**Feature importance instability** has emerged in F7: x2's importance drops 52% and x5's drops 63% when the best point is removed. This means these dimensions' apparent importance is driven by a single data point rather than genuine function structure. I flag this and only trust model consensus on dimensions with robust importance.

For F3 (3D, 17 points), no model beats baseline despite x3 showing r=-0.59. The correlation is inflated by one outlier (pt7, Y=-0.40) — without it, r drops to -0.21. More data hasn't helped because the function's structure in x1 and x2 remains too noisy to capture.

Interestingly, overfitting hasn't been a major issue because LOOCV naturally penalises it. Models that overfit show poor LOO scores and get filtered out. The bigger problem is underfitting — in 5D+ spaces, 20-30 points simply isn't enough for flexible models to learn the landscape.

### Q5: How does this prepare you to think like a data scientist?

This exercise teaches that **the method matters less than knowing when to trust it**. I run the same GridSearchCV pipeline on all functions, but the decision of whether to follow the model, use a centroid, or try something creative depends entirely on validation performance. A 70% improvement model gets trusted; a -5% model gets ignored — regardless of how sophisticated it is.

The F1 classifier insight exemplifies real-world data science thinking: when the obvious approach fails (regression on near-zero data), reframe the problem. Splitting an impossible regression into two solvable sub-problems (classification + log-space regression) is exactly the kind of creative decomposition that real projects demand.

Finally, the per-function strategy selection mirrors real-world ML: different datasets need different approaches, and the ability to diagnose why a model fails (outlier inflation, boundary extrapolation, insufficient data density) matters more than knowing the latest algorithm.

---

# Submission Answer

**Q1: How has your query strategy changed?**

I moved from a uniform pipeline to function-specific strategies. For functions with strong models (F4 SVR at 70%, F6 SVR at 55%), I trust the dominant model. For functions where models disagree on some dimensions (F7, F8), I built a hybrid: centroid for uncertain dimensions, model consensus only where all models agree (spread < 0.2). For F1, where regression fails completely (all Y values near zero), I split the problem into classification (SVM predicting sign of Y, 75% accuracy) plus log-space SVR for signal strength — neither works alone, but combined they triangulate a search target. I also learned to exclude linear models (Ridge, SVR) when they extrapolate to boundary corners.

**Q2: Exploration vs exploitation?**

Mostly exploitation, but calibrated by model reliability. F5 has improved three consecutive weeks (984 → 1089 → 1207) using model-guided perturbations — clear case for exploitation. F4 and F8 also exploit strong models (70% and 80% improvement). F1 is informed exploration: using a classifier to explore the region most likely to contain the peak, rather than random space-filling. F3 uses centroid (no model beats baseline). The key insight: exploitation only works where models have earned trust through cross-validation.

**Q3: How would SVMs change your approach?**

I already implemented this for F1. An SVM classifier predicting sign(Y) achieves 75% LOO accuracy on data where regression completely fails. Combined with log-space SVR, it identifies the most promising search region. A soft-margin SVM would help F5 and F8 by classifying "promising" vs "unpromising" regions with a performance threshold. An RBF kernel SVM would suit F4's confirmed quadratic structure without manual polynomial feature engineering.

**Q4: Model limitations as data grows?**

Three issues emerged. First, linear models extrapolate to boundaries — Ridge consistently suggests corner points, now filtered from ensembles. Second, feature importance instability: F7's x2 importance drops 52% without the best point, meaning it's driven by one data point, not genuine structure. Third, in F3 the dominant correlation (x3, r=-0.59) collapses to r=-0.21 without one outlier — more data hasn't helped because the signal was never real. Overfitting is less of an issue since LOOCV naturally penalises it; the bigger problem is underfitting in high-dimensional spaces with sparse data.

**Q5: How does this prepare you for data science?**

It teaches that knowing when to trust a model matters more than the model itself. I run the same pipeline everywhere, but a 70% improvement model gets trusted while a -5% model gets ignored. The F1 classifier insight captures real-world thinking: when the obvious approach fails, reframe the problem creatively. And per-function strategy selection mirrors real ML projects — different datasets need different approaches, and diagnosing why a model fails matters more than knowing the latest algorithm.

---

# Submission Answer (HTML)

```html
<p><strong>Q1</strong></p>
<p>I moved from a uniform pipeline to function-specific strategies. For functions with strong models (F4 SVR at 70%, F6 SVR at 55%), I trust the dominant model. For functions where models disagree on some dimensions (F7, F8), I built a hybrid: centroid for uncertain dimensions, model consensus only where all models agree (spread &lt; 0.2). For F1, where regression fails completely (all Y values near zero), I split the problem into classification (SVM predicting sign of Y, 75% accuracy) plus log-space SVR for signal strength — neither works alone, but combined they triangulate a search target. I also learned to exclude linear models (Ridge, SVR) when they extrapolate to boundary corners.</p>

<p><strong>Q2</strong></p>
<p>Mostly exploitation, but calibrated by model reliability. F5 has improved three consecutive weeks using model-guided perturbations. F4 and F8 also exploit strong models. F1 is informed exploration: using a classifier to explore the region most likely to contain the peak, rather than random space-filling. F3 uses centroid. The key insight: exploitation only works where models have earned trust through cross-validation.</p>

<p><strong>Q3</strong></p>
<p>I already implemented this for F1. An SVM classifier predicting sign(Y) achieves 75% LOO accuracy on data where regression completely fails. Combined with log-space SVR, it identifies the most promising search region. A soft-margin SVM would help F5 and F8 by classifying "promising" vs "unpromising" regions with a performance threshold. An RBF kernel SVM would suit F4's confirmed quadratic structure without manual polynomial feature engineering.</p>

<p><strong>Q4</strong></p>
<p>Three issues emerged. First, linear models extrapolate to boundaries. Ridge consistently suggests corner points, now filtered from ensembles. Second, feature importance instability: F7's x2 importance drops 52% without the best point, meaning it's driven by one data point, not genuine structure. Third, in F3 the dominant correlation (x3, r=-0.59) collapses to r=-0.21 without one outlier, more data hasn't helped because the signal was never real. Overfitting is less of an issue since LOOCV naturally penalises it. The bigger problem is underfitting in high-dimensional spaces with sparse data.</p>

<p><strong>Q5</strong></p>
<p>It teaches that knowing when to trust a model matters more than the model itself. I run the same pipeline everywhere, but a 70% improvement model gets trusted while a -5% model gets ignored. The F1 classifier insight captures real-world thinking: when the obvious approach fails, reframe the problem creatively. And per-function strategy selection mirrors real ML projects, different datasets need different approaches, and diagnosing why a model fails matters more than knowing the latest algorithm.</p>
```
