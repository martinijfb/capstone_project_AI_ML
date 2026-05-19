# Week 08 Reflection: Module 19 (LLMs and Prompting)

## Background notes (not for submission)

- Word limit: 700 (portal). Seven LLM-themed prompts mapped onto BBO concepts.
- The honest framing: I don't use LLMs to generate queries. The prompts make most sense as analogies between LLM decoding settings and BBO decisions (peer post confirms this is the expected approach).
- Anchored on: F1 gradient-climb (structured "few-shot"), F3 outlier-as-irrelevant-context, F5/F7/F8 TuRBO multi-kernel TS as "high temperature" decoding, validate-then-trust pipeline as hallucination prevention.

## Submission (plain text, ~680 words)

**Q1**

The closest analog to zero-shot vs few-shot is space-filling vs model-guided queries. Zero-shot has no in-context examples; my balanced-Voronoi candidates ignore prior bests and pick the most under-sampled region. Few-shot passes examples in; every regressor I fit (KNN, RF, GB, GP family) is "few-shot", building a response from observed data. This week's cleanest structured prompt was on F1, where the standard ensemble had stopped extracting signal. I stripped it to two examples (a known positive and its nearest known negative) and one instruction: gradient-climb along the +/− jump. Simplifying made the decision defensible in a way the noisy ensemble no longer did.

**Q2**

Step magnitude is my temperature analog. This week some functions used tiny refinement steps (low temperature, deterministic) and others used TuRBO trust-region exploration (much higher temperature). top-p maps to TuRBO's trust-region length, controlling the fraction of input space the next candidate can come from. top-k maps to the number of valid interior models surviving boundary and outlier filters, about half of the ten I fit. max-tokens is the implicit per-dim step cap. Tight settings refine but cannot escape a plateau; loose settings explore but risk falling off.

**Q3**

My "tokens" are continuous numbers in [0, 1). Token-boundary effects show up when models extrapolate beyond observed-good values. The boundary-rejection rule drops any candidate with a dim < 0.02 or > 0.98 unless three or more non-linear models agree on the edge AND the per-dim correlation supports it. This week the rule fired on F5, where top points sit near the upper edge on three dimensions; the framework clipped each to the top-5 maximum rather than letting models extrapolate. No truncation issues observed. The "token count" analog is training-point count: F1/F2 are data-starved at 17; F8 sits comfortably at 47.

**Q4**

F1 and F2 sit at 17 points. F1 illustrates "attention focusing on irrelevant context" via its sign classifier (an SVM I train on whether Y is positive or negative): a recent query added a vanishing-magnitude point that was numerically near zero but had to be treated as a real negative. Classifier LOO accuracy dropped 87% to 81% to 71% across three weeks. Each new low-magnitude point shifts the boundary disproportionately. F2 shows "diminishing returns": a very small step from the current best caused a meaningful Y drop, suggesting either evaluation noise or a peak so sharp the new sample confuses the GP fit rather than refining it.

**Q5**

Validate-then-trust is central. No model influences a query unless its LOOCV RMSE beats the baseline; that alone catches most fabrications. On top: boundary-rejection, outlier-suggestion filter (drop models whose argmax is a spatial outlier from the rest), and reading the prior week's suggestions notes before deciding (retrieval of prior relevant information). The clearest hallucination catch this week was on F3: a single point at an extreme value of one input was driving the apparent correlation between that input and Y from strongly negative to essentially zero once removed. Several smooth GP variants were hallucinating an attractive peak in an unexplored mid-range entirely because of that point. Removing it, the same models converged on a totally different region.

**Q6**

With larger data I would replace the deterministic ensemble argmax with principled acquisition functions: GP-UCB with a learned β schedule, qLogNEI from BoTorch, expected improvement. The multi-kernel Thompson sampling I added this week scales by adding kernels. For thousands of points, approximate GPs (sparse variational, KISS-GP) become necessary because full GP is cubic. For more complex models, Bayesian hyperparameter optimisation over the surrogate (Optuna, BoTorch) is the next layer.

**Q7**

The capstone constraint (one query per function per week, no gradient access, no functional form, real-time decisions) is a hard version of the practitioner trade-off. This week I had three risk levels on TuRBO-style bets: one continuation of a validated experiment, one where the framework rule mandated the switch after consecutive regressions, and one where I overrode the conservative default for bolder exploration. A practitioner triages the same way: which systems are working, which need intervention, which deserve a bold experiment. The discipline that transfers is treating every modelling choice as something that has to survive cross-validation before influencing a decision.

## Submission (HTML version for portal)

```html
<p><strong>Q1</strong></p>
<p>The closest analog to zero-shot vs few-shot is space-filling vs model-guided queries. Zero-shot has no in-context examples; my balanced-Voronoi candidates ignore prior bests and pick the most under-sampled region. Few-shot passes examples in; every regressor I fit (KNN, RF, GB, GP family) is "few-shot", building a response from observed data. This week's cleanest structured prompt was on F1, where the standard ensemble had stopped extracting signal. I stripped it to two examples (a known positive and its nearest known negative) and one instruction: gradient-climb along the +/− jump. Simplifying made the decision defensible in a way the noisy ensemble no longer did.</p>

<p><strong>Q2</strong></p>
<p>Step magnitude is my temperature analog. This week some functions used tiny refinement steps (low temperature, deterministic) and others used TuRBO trust-region exploration (much higher temperature). top-p maps to TuRBO's trust-region length, controlling the fraction of input space the next candidate can come from. top-k maps to the number of valid interior models surviving boundary and outlier filters, about half of the ten I fit. max-tokens is the implicit per-dim step cap. Tight settings refine but cannot escape a plateau; loose settings explore but risk falling off.</p>

<p><strong>Q3</strong></p>
<p>My "tokens" are continuous numbers in [0, 1). Token-boundary effects show up when models extrapolate beyond observed-good values. The boundary-rejection rule drops any candidate with a dim &lt; 0.02 or &gt; 0.98 unless three or more non-linear models agree on the edge AND the per-dim correlation supports it. This week the rule fired on F5, where top points sit near the upper edge on three dimensions; the framework clipped each to the top-5 maximum rather than letting models extrapolate. No truncation issues observed. The "token count" analog is training-point count: F1/F2 are data-starved at 17; F8 sits comfortably at 47.</p>

<p><strong>Q4</strong></p>
<p>F1 and F2 sit at 17 points. F1 illustrates "attention focusing on irrelevant context" via its sign classifier (an SVM I train on whether Y is positive or negative): a recent query added a vanishing-magnitude point that was numerically near zero but had to be treated as a real negative. Classifier LOO accuracy dropped 87% to 81% to 71% across three weeks. Each new low-magnitude point shifts the boundary disproportionately. F2 shows "diminishing returns": a very small step from the current best caused a meaningful Y drop, suggesting either evaluation noise or a peak so sharp the new sample confuses the GP fit rather than refining it.</p>

<p><strong>Q5</strong></p>
<p>Validate-then-trust is central. No model influences a query unless its LOOCV RMSE beats the baseline; that alone catches most fabrications. On top: boundary-rejection, outlier-suggestion filter (drop models whose argmax is a spatial outlier from the rest), and reading the prior week's suggestions notes before deciding (retrieval of prior relevant information). The clearest hallucination catch this week was on F3: a single point at an extreme value of one input was driving the apparent correlation between that input and Y from strongly negative to essentially zero once removed. Several smooth GP variants were hallucinating an attractive peak in an unexplored mid-range entirely because of that point. Removing it, the same models converged on a totally different region.</p>

<p><strong>Q6</strong></p>
<p>With larger data I would replace the deterministic ensemble argmax with principled acquisition functions: GP-UCB with a learned β schedule, qLogNEI from BoTorch, expected improvement. The multi-kernel Thompson sampling I added this week scales by adding kernels. For thousands of points, approximate GPs (sparse variational, KISS-GP) become necessary because full GP is cubic. For more complex models, Bayesian hyperparameter optimisation over the surrogate (Optuna, BoTorch) is the next layer.</p>

<p><strong>Q7</strong></p>
<p>The capstone constraint (one query per function per week, no gradient access, no functional form, real-time decisions) is a hard version of the practitioner trade-off. This week I had three risk levels on TuRBO-style bets: one continuation of a validated experiment, one where the framework rule mandated the switch after consecutive regressions, and one where I overrode the conservative default for bolder exploration. A practitioner triages the same way: which systems are working, which need intervention, which deserve a bold experiment. The discipline that transfers is treating every modelling choice as something that has to survive cross-validation before influencing a decision.</p>
```
