# Week 02 — Reflection

## Full Reflection (detailed version)

### Q1: What was the main change in your strategy compared to last week?

The biggest change was switching from manual model configurations to sklearn's `GridSearchCV` with LOO cross-validation. In week 01 I tested individual model configs one by one — this week I let GridSearchCV systematically search hyperparameter spaces (e.g. SVR C=[0.1, 1, 10, 100] × gamma=[scale, auto]). This found significantly better configs: F6's SVR jumped from C=1 (30% improvement) to C=100 (51%), and F7's KNN improved from 5.3% to 31.5%.

The second change was learning from week 01 failures. Three functions failed to improve, and each failure taught something specific: F3's centroid failed because the x3 correlation was inflated by an outlier, F5's GP perturbation went downhill on an extremely steep gradient, and F7's x6 push to 0.90 was counterproductive. Each required a different corrective strategy this week.

### Q2: Exploration vs exploitation? Trade-offs?

It was function-specific. F4 and F8 improved in week 01, so I exploited near the new best points using GP and SVR suggestions. F2 also shifted to exploitation — week 01 confirmed bimodality, so I now perturb near the better peak rather than probing between them.

F1 remains exploratory — the directional perturbation from pt2 is informed by the two negative points we found, but it's still searching for the hidden source. F3 is exploratory too, following the only model (GB depth=2) that beat baseline on clean data.

F5 is an interesting case: the week 01 perturbation went downhill, so I'm trying the exact opposite direction. It's exploitation in the sense that I'm using gradient information, but it's a gamble since I don't know if the function is symmetric around pt15.

### Q3: Have participant strategies or class discussions influenced your approach?

Not directly this week. However, reviewing classmates' reflections from week 01 confirmed that F1 is universally difficult (everyone found near-zero outputs) and that adaptive kappa strategies are common. Sterling Drake's approach of tuning kappa per function based on week 01 results is similar in spirit to my per-function model selection, though I use different model families rather than adjusting a single parameter.

### Q4: Linear/logistic regression assumptions — which would you violate?

For F4, a linear regression would violate several assumptions. In week 01 I tested polynomial features and found that degree-2 features achieved 85% LOOCV improvement — far better than plain linear regression at 23%. This confirms significant non-linearity: the function has quadratic interactions (x1², x1·x2, etc.) that a linear model misses. The negative correlations (x1: -0.53, x2: -0.45, x4: -0.50) suggest a downward slope, but the actual optimum is in the interior at moderate values — a linear model would extrapolate to the boundary.

For F5, the output spans from 0.11 to 1088 — three orders of magnitude. A linear model would struggle with this range without log-transformation, and the residuals would be heavily heteroscedastic (larger errors at higher Y values).

### Q5: Linear regions or decision boundaries?

F8 has the most plausible linear relationship. Features x1 (r=-0.64) and x3 (r=-0.65) both show strong, stable negative correlations that persist even when the best point is removed (feature importance drops only 8% and 5% respectively). In the low-x1/low-x3 region, the output clusters above 9.0, which could form a rough decision boundary.

A logistic classifier with a threshold at Y=9.0 would likely achieve decent accuracy on F8's training data, since the x1+x3 subspace separates high from low outputs reasonably well. However, with 41 points in 8D, the classifier would be unreliable on dimensions x4-x8 where correlations are near zero.

For F2, the bimodality I confirmed this week (valley at x2≈0.53 between peaks at x2≈0.12 and x2≈0.93) means a logistic classifier could work if framed as "high x2 peak vs low x2 peak" — but the two peaks have different Y values, so the boundary would be noisy.

### Q6: Feature effects and interpretability?

Yes, examining per-feature correlations and feature importance was central to my approach this week. Two specific findings changed my strategy:

For F3, removing outlier pt6 caused the x3 correlation to collapse from r=-0.56 to r=-0.11 — what appeared to be the dominant signal was actually driven by a single extreme data point. This is why the week 01 centroid (which relied on the x3 trend) failed. The feature-level analysis directly prevented me from repeating the same mistake.

For F7, the feature importance robustness check showed x1's Random Forest importance dropped 69% when the best point was removed. Combined with the failed x6=0.90 push, this told me the week 01 analysis had overfitted to individual point characteristics rather than genuine function structure.

---

# Submission Answer

**Q1: Main strategy change?**

Two changes. First, I replaced manual model configs with sklearn's `GridSearchCV` and LOO CV — this systematically searches hyperparameter spaces and found significantly better models (e.g. F6's SVR went from C=1 at 30% improvement to C=100 at 51%). Second, I adapted each function's strategy based on week 01 failures rather than repeating the same approach.

**Q2: Exploration vs exploitation?**

My strategies fell into three groups:

*Exploitation* (F4, F6, F8): These improved in week 01, so I followed the best GridSearchCV model — GP for F4/F8, SVR for F6. The trade-off is risking a local maximum, but validated models give enough confidence.

*Exploration* (F1, F3): Too little signal for models. F1 steps away from two negative points toward the likely source. F3 follows the only model that beat baseline after cleaning out an outlier.

*Correction* (F2, F5, F7): Week 01 told me which direction was wrong, so I adjusted. F5 mirrors a failed perturbation, F7 returns near the original best after an aggressive push backfired.

**Q3: Peer influence?**

Yes — reading classmates' reflections helped me understand the general consensus and validate my approach. It confirmed that F1 is universally difficult (everyone found near-zero outputs) and that most people are using adaptive exploration-exploitation strategies. Seeing that several classmates use GP+UCB with varying kappa per function gave me confidence that my per-function model selection is a reasonable alternative — I achieve a similar adaptive balance but through different model families rather than tuning a single parameter.

**Q4: Linear regression assumptions violated?**

For F4, I tested polynomial features and LOOCV improvement jumped from 23% (linear) to 85% (degree=2) — confirming strong non-linear interactions (x1², x1·x2). A linear model would also extrapolate to boundary values while the true optimum is interior. For F5, the output spans three orders of magnitude (0.11 to 1088), meaning prediction errors would be much larger at high Y values than low ones — unequal variance.

**Q5: Linear regions or decision boundaries?**

F8 is the best candidate. Features x1 (r=-0.64) and x3 (r=-0.65) show robust correlations that persist when the best point is removed. A logistic classifier thresholding at Y≈9.0 could work in the x1-x3 subspace. For F2, confirmed bimodality in x2 could define a boundary, but the two peaks have different Y values making it noisy.

**Q6: Feature effects useful?**

For F3, removing one outlier collapsed the x3 correlation from r=-0.56 to r=-0.11 — the week 1 "dominant signal" was a single-point artifact. For F7, x1's Random Forest importance dropped 69% without the best point. Both findings directly changed my strategy and prevented repeating week 1's mistakes.

---

# Submission Answer (HTML)

```html
<p><strong>Q1: Main strategy change?</strong></p>
<p>I replaced manual model configs with sklearn's GridSearchCV and LOOCV, this systematically searches hyperparameter spaces and found significantly better models (e.g. F6's SVR went from C=1 at 30% improvement to C=100 at 51%). I also adapted each function's strategy based on week 01 failures rather than repeating the same approach.</p>

<p><strong>Q2: Exploration vs exploitation?</strong></p>
<p>My strategies fell into three groups:</p>
<p><em>Exploitation</em> (F4, F6, F8): These improved in week 01, so I followed the best GridSearchCV model — GP for F4/F8, SVR for F6. The trade-off is risking a local maximum, but validated models give enough confidence.</p>
<p><em>Exploration</em> (F1, F3): Too little signal for models. F1 steps away from two negative points toward the likely source. F3 follows the only model that beat baseline after cleaning out an outlier.</p>
<p><em>Correction</em> (F2, F5, F7): Week 01 told me which direction was wrong, so I adjusted. F5 mirrors a failed perturbation, F7 returns near the original best after an aggressive push backfired.</p>

<p><strong>Q3: Peer influence?</strong></p>
<p>Yes, reading classmates' reflections helped me understand the general consensus and validate my approach. It confirmed that F1 is universally difficult and that most people use adaptive exploration-exploitation strategies. Seeing that several classmates use GP+UCB with varying kappa gave me confidence that my per-function model selection is a reasonable alternative. I achieve a similar balance but through different model families.</p>

<p><strong>Q4: Linear regression assumptions violated?</strong></p>
<p>For F4, I tested polynomial features and LOOCV improvement jumped from 23% (linear) to 85% (degree=2), confirming strong non-linear interactions. For F5, the output spans three orders of magnitude (0.11 to 1088), meaning prediction errors would be much larger at high Y values than low ones — unequal variance.</p>

<p><strong>Q5: Linear regions or decision boundaries?</strong></p>
<p>F8 is the best candidate. Features x1 (r=-0.64) and x3 (r=-0.65) show robust correlations that persist when the best point is removed. A logistic classifier thresholding at Y≈9.0 could work in the x1-x3 subspace. For F2, bimodality in x2 could define a boundary, but the two peaks have different Y values making it noisy.</p>

<p><strong>Q6: Feature effects useful?</strong></p>
<p>For F3, removing one outlier collapsed the x3 correlation from r=-0.56 to r=-0.11, the week 1 "dominant signal" was a single-point artifact. For F7, x1's Random Forest importance dropped 69% without the best point. Both findings directly changed my strategy and prevented repeating week 1's mistakes.</p>
```
