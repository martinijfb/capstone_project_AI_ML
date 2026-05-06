# Week 07 Reflection: Module 18 (Hyperparameter Tuning)

## Background notes (not for submission)

- Word limit: 700 (portal). Six prompts about hyperparameter tuning.
- Anchored on: GridSearchCV with LOOCV across 7 sklearn families, Matern ν as a CV-chosen hyperparameter (added in W6), NN regularisation variants (plain/dropout/wd/ensemble) × H ∈ {16, 32}, TuRBO trust region L self-adapting, GP-UCB β with kappa decay schedule.

## Submission (plain text, ~690 words)

**Q1**

Each model family has its own knobs. For Ridge it is the regularisation alpha. For KNN it is n_neighbors and the weighting scheme. For Random Forest it is max_depth and n_estimators. For SVR it is C and gamma. For Gradient Boosting it is max_depth, n_estimators and learning_rate. For the Gaussian Processes I now treat the Matern smoothness parameter ν as a hyperparameter with three values (0.5, 1.5 and 2.5), where previously I had ν fixed at 2.5. For the neural network surrogates I tune over hidden width (16 or 32) and four regularisation variants: plain, dropout, weight decay, and a five-seed ensemble. I prioritised the standard interpretable knobs because they are the ones I could justify if asked, and Matern ν because the kernel smoothness had felt like an arbitrary choice in earlier rounds.

**Q2**

The biggest change came from letting cross-validation pick the kernel smoothness instead of fixing it. The result reshuffled my model rankings on four of the eight functions. On F4, F5, F6 and F7 the rougher Matern variants either lead the leaderboard or tie at the top. F5's leading model has been Matern ν=0.5 for three weeks running. In earlier rounds I would have submitted queries informed by a smoother kernel and missed that signal entirely.

**Q3**

For all sklearn models I use exhaustive grid search with leave-one-out cross-validation. The grids are small (three to twelve configurations per family), so the grid search costs less than the model fits themselves. Random search would explore more of the hyperparameter space, but at 15 to 46 data points the grids I have are already saturated. For the trust-region method I added this week, the trust-region length L is itself a hyperparameter, but it adapts automatically: it doubles after three consecutive successes and halves after four consecutive failures. The optimiser is tuning itself. For the upper-confidence-bound acquisition I tested as a second opinion, the exploration coefficient beta is set by a manual schedule that decays from 2.0 in week 1 to 0.5 by week 12. The trade-off there is that the schedule is my judgement about how the project should mature, not something learned from data.

**Q4**

Three things. First, F1 has 16 points and no hyperparameter choice rescues any model: every variant fails to beat the baseline of predicting the mean. Tuning makes this honest. Second, the same hyperparameter prefers different values on different functions. Matern ν=0.5 wins on F4 through F7 but loses to ν=2.5 on F8. There is no single "best" Gaussian Process configuration across the project. Third, on F2 the cross-validation spread of model suggestions has narrowed to 0.008 across both dimensions and the latest weekly step is 0.003. The model is converging, and further tuning will not produce more information until I have more data.

**Q5**

With more data the search space widens and full Bayesian optimisation over hyperparameters becomes worthwhile. Tools like BoTorch, Optuna and scikit-optimize implement this cleanly. For the remaining capstone weeks, the highest-value upgrade I have planned is on the acquisition side: testing GP-UCB with a learned beta schedule, or expected improvement, instead of the deterministic ensemble argmax. For deep-network projects where each hyperparameter combination costs minutes or hours rather than seconds, Hyperband and successive-halving would matter more than they do here.

**Q6**

Real-world systems rarely come with clean held-out sets. Medical, industrial and financial systems evolve, and the test distribution is whatever happens after deployment. The discipline of validate-then-trust, requiring leave-one-out evidence before letting any model influence a decision, is the skill that transfers. The boundary-rejection rule is the same scepticism applied: a model's confident extrapolation past observed data is not a signal, it is the model being asked a question its training set could not answer. Tuning a hyperparameter is one small example of the larger discipline of treating every modelling choice as something to be defended with evidence.

## Submission (HTML version for portal)

```html
<p><strong>Q1</strong></p>
<p>Each model family has its own knobs. For Ridge it is the regularisation alpha. For KNN it is n_neighbors and the weighting scheme. For Random Forest it is max_depth and n_estimators. For SVR it is C and gamma. For Gradient Boosting it is max_depth, n_estimators and learning_rate. For the Gaussian Processes I now treat the Matern smoothness parameter ν as a hyperparameter with three values (0.5, 1.5 and 2.5), where previously I had ν fixed at 2.5. For the neural network surrogates I tune over hidden width (16 or 32) and four regularisation variants: plain, dropout, weight decay, and a five-seed ensemble. I prioritised the standard interpretable knobs because they are the ones I could justify if asked, and Matern ν because the kernel smoothness had felt like an arbitrary choice in earlier rounds.</p>

<p><strong>Q2</strong></p>
<p>The biggest change came from letting cross-validation pick the kernel smoothness instead of fixing it. The result reshuffled my model rankings on four of the eight functions. On F4, F5, F6 and F7 the rougher Matern variants either lead the leaderboard or tie at the top. F5's leading model has been Matern ν=0.5 for three weeks running. In earlier rounds I would have submitted queries informed by a smoother kernel and missed that signal entirely.</p>

<p><strong>Q3</strong></p>
<p>For all sklearn models I use exhaustive grid search with leave-one-out cross-validation. The grids are small (three to twelve configurations per family), so the grid search costs less than the model fits themselves. Random search would explore more of the hyperparameter space, but at 15 to 46 data points the grids I have are already saturated. For the trust-region method I added this week, the trust-region length L is itself a hyperparameter, but it adapts automatically: it doubles after three consecutive successes and halves after four consecutive failures. The optimiser is tuning itself. For the upper-confidence-bound acquisition I tested as a second opinion, the exploration coefficient beta is set by a manual schedule that decays from 2.0 in week 1 to 0.5 by week 12. The trade-off there is that the schedule is my judgement about how the project should mature, not something learned from data.</p>

<p><strong>Q4</strong></p>
<p>Three things. First, F1 has 16 points and no hyperparameter choice rescues any model: every variant fails to beat the baseline of predicting the mean. Tuning makes this honest. Second, the same hyperparameter prefers different values on different functions. Matern ν=0.5 wins on F4 through F7 but loses to ν=2.5 on F8. There is no single "best" Gaussian Process configuration across the project. Third, on F2 the cross-validation spread of model suggestions has narrowed to 0.008 across both dimensions and the latest weekly step is 0.003. The model is converging, and further tuning will not produce more information until I have more data.</p>

<p><strong>Q5</strong></p>
<p>With more data the search space widens and full Bayesian optimisation over hyperparameters becomes worthwhile. Tools like BoTorch, Optuna and scikit-optimize implement this cleanly. For the remaining capstone weeks, the highest-value upgrade I have planned is on the acquisition side: testing GP-UCB with a learned beta schedule, or expected improvement, instead of the deterministic ensemble argmax. For deep-network projects where each hyperparameter combination costs minutes or hours rather than seconds, Hyperband and successive-halving would matter more than they do here.</p>

<p><strong>Q6</strong></p>
<p>Real-world systems rarely come with clean held-out sets. Medical, industrial and financial systems evolve, and the test distribution is whatever happens after deployment. The discipline of validate-then-trust, requiring leave-one-out evidence before letting any model influence a decision, is the skill that transfers. The boundary-rejection rule is the same scepticism applied: a model's confident extrapolation past observed data is not a signal, it is the model being asked a question its training set could not answer. Tuning a hyperparameter is one small example of the larger discipline of treating every modelling choice as something to be defended with evidence.</p>
```
