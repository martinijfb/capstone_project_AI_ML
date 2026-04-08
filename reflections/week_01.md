# Week 01 — Reflection

## What was the main principle or heuristic used to decide on each query point?

Rather than applying a single method across all functions, I tailored the approach per function based on data availability, dimensionality, and whether surrogate models could be trusted. My core framework was: fit multiple models, validate with Leave-One-Out Cross-Validation (LOOCV RMSE), and only trust suggestions from models that beat the baseline (predicting the mean) and suggest interior points (not near boundaries).

For **F1** (2D, 10 points), all outputs were effectively zero — no signal for any model to learn from. I used a computational geometry approach: the Largest Empty Circle via Voronoi diagram, constrained to [0,1]², to find the point maximising minimum distance to all samples. Pure exploration.

For **F2** (2D) and **F3** (3D), I used a Y-weighted centroid of the top 4 performers. In F2, this probed the midpoint between two peaks with similar x1 but opposite x2 — testing bimodality. In F3, I tested 7 sklearn models (Linear, Ridge, KNN, Random Forest, SVR, GP) with LOOCV and none beat the baseline, so models were rejected in favour of the data-anchored centroid.

For **F4** (4D, 30 points), SVR with RBF kernel (C=10) achieved 63% LOOCV improvement over baseline — the first function where models genuinely worked. I also tested polynomial features on linear regression, confirming the function has quadratic structure. Exploitation-focused.

For **F5** (4D, unimodal), the GP suggestion landed 0.027 from the best known point — essentially a smart perturbation near the peak. A feature importance robustness check revealed that x3's apparent importance was entirely inflated by one outlier; x4 was the true driver.

For **F6** (5D), SVR (C=1) achieved 30% improvement. For **F7** (6D), I ran a grid search across 17 model configurations — only 6 beat baseline by tiny margins. I used the centroid for x1-x5 but pushed x6 to 0.90, the only dimension with strong model consensus.

For **F8** (8D, 40 points), GP achieved 78.5% improvement — the strongest model across all functions. However, I only trusted GP on x1 and x3 (strong correlation, robust feature importance, model consensus) and kept centroid values for x4-x8 where models disagreed.

## Which function(s) were most challenging to query and why?

**F1** was the hardest. With all 10 outputs at ~0 (magnitudes from 1e-124 to 1e-16), there was no signal for any surrogate model. I tried a log-magnitude gradient analysis — treating log₁₀|Y| as a proxy signal — but with one meaningful data point it was unreliable. The radiation detection analogy implies a very narrow peak, and none of the initial samples were near it.

**F3** was also challenging. Despite having 15 points and a visible x3 correlation (r=-0.57), no model beat the baseline under LOOCV. One outlier (pt6, Y=-0.40, 3.2σ from the next worst) distorted every model. Prior knowledge of the output scale or function smoothness would have helped for both.

## How do you plan to adjust your strategy in future rounds?

My strategy depends on the results returned. For **F1**, if the query returns non-zero signal, I'll switch from space-filling to log-gradient analysis or GP+UCB. If still zero, I'll try Latin Hypercube Sampling for systematic coverage. For **F4** and **F8**, where models worked well, I'll re-fit with the additional data point and trust the updated model. For **F5**, I'll continue exploiting near the peak but test whether x4 alone drives performance. For **F3** and **F7**, where models failed, I'll re-evaluate whether the extra data point tips any model past the baseline. As more data accumulates, I expect to shift gradually from centroid-based approaches to model-guided ones, and from exploration to exploitation — but only where models earn that trust through validation.

---

# Submission Answer

**What was the main principle or heuristic used to decide on each query point?**

I didn't use a single method for all functions — I picked the approach based on whether models could actually be trusted for each one. My process was: fit several models (Ridge, KNN, RF, SVR, GP), validate each with LOOCV, and only use their suggestions if they beat the baseline and didn't hit boundaries.

For F1, outputs were all essentially zero, so no model could learn anything. I went with a Voronoi-based space-filling approach instead — pure exploration. For F2 and F3, I used Y-weighted centroids of the top performers. In F3 specifically, I tested 7 models and none beat baseline, so I rejected them and stuck with the centroid. For F4, SVR worked well (63% LOOCV improvement), so I trusted its suggestion. F5 is unimodal — GP suggested a point barely 0.03 away from the best known point, which is the right move for a single peak. For F6, SVR again (30% improvement). F7 was tricky — ran 17 model configs and only 6 barely beat baseline, but they all agreed on one thing: high x6. So I kept the centroid for most dimensions and only pushed x6 higher. For F8, GP was excellent (78.5% improvement) but I only trusted it on x1 and x3, where correlations were strong and feature importance was robust. Centroid for the rest.

**Which function(s) were most challenging to query and why?**

F1 by far. Every output is near zero (magnitudes from 1e-124 to 1e-16), so there's basically no signal. I tried treating log|Y| as a proxy signal to find a directional trend, but with effectively one non-trivial data point it wasn't reliable. The function seems to have an extremely narrow peak and we haven't landed near it yet. Knowing something about the output scale or peak width would have helped enormously.

F3 was also frustrating — there's a visible correlation on x3 (r=-0.57) but no model could generalise with just 15 points and an outlier distorting everything.

**How do you plan to adjust your strategy in future rounds?**

It depends on what comes back. If F1 returns a non-zero value, that changes everything — I'd switch to GP+UCB immediately. If it's still zero, I'll try Latin Hypercube Sampling for more systematic coverage. For F4 and F8 where models worked, I'll just re-fit with the new data and follow the updated suggestion. For F3 and F7, I'll check whether the extra point finally lets a model beat baseline. The general plan is to move from exploration to exploitation over time, but only on functions where I have enough confidence in the surrogate model to justify it.

---

# Submission Answer (HTML)

```html
<p><strong>What was the main principle or heuristic used to decide on each query point?</strong></p>

<p>I didn't use a single method for all functions. I picked the approach based on whether models could actually be trusted for each one. My process was: fit several models (Ridge, KNN, RF, SVR, GP), validate each with LOOCV, and only use their suggestions if they beat the baseline and didn't hit boundaries.</p>

<p>For F1, outputs were all essentially zero, so no model could learn anything. I went with a Voronoi-based space-filling approach instead, pure exploration. For F2 and F3, I used Y-weighted centroids of the top performers. In F3 specifically, I tested 7 models and none beat baseline, so I rejected them and stuck with the centroid. For F4, SVR worked well, so I trusted its suggestion. F5 is unimodal, GP suggested a point barely 0.03 away from the best known point. For F6, SVR again. F7 was tricky, ran 17 model configs and only 6 barely beat baseline, but they all agreed on one thing: high x6. So I kept the centroid for most dimensions and only pushed x6 higher. For F8, GP was excellent, but I only trusted it on x1 and x3, where correlations were strong and feature importance was robust. Centroid for the rest.</p>

<p><strong>Which function(s) were most challenging to query and why?</strong></p>

<p>F1 by far. Every output is near zero, so there's basically no signal. I tried treating log|Y| as a proxy signal to find a directional trend, but with effectively one non-trivial data point it wasn't reliable. The function seems to have an extremely narrow peak. Knowing something about the output scale or peak width would have helped enormously.</p>

<p>F3 was also frustrating, there's a visible correlation on x3 (r=-0.57) but no model could generalise with just 15 points and an outlier distorting everything.</p>

<p><strong>How do you plan to adjust your strategy in future rounds?</strong></p>

<p>It depends on what comes back. If F1 returns a non-zero value, that changes everything, I'd switch to GP+UCB immediately. If it's still zero, I'll try Latin Hypercube Sampling for more systematic coverage. For F4 and F8 where models worked, I'll just re-fit with the new data and follow the updated suggestion. For F3 and F7, I'll check whether the extra point finally lets a model beat baseline. The general plan is to move from exploration to exploitation over time, but only on functions where I have enough confidence in the surrogate model to justify it.</p>
```
