# Week 04 — Capstone 15.2 Reflection: Hyperparameter tuning

## Portal prompts (3)

1. **Hyperparameter effects**: What are some key hyperparameters you have used or observed in neural networks? How did changing these affect your model's convergence, stability or performance?
2. **Discrete vs continuous**: Categorise the hyperparameters you've worked with. Which are continuous and which are discrete? How might the type of hyperparameter influence the method used to tune it?
3. **Application to the capstone**: If you are using or considering a neural network as a surrogate model in your capstone, how will your understanding of hyperparameter tuning influence your next set of decisions? Could you apply your BBO approach to improve your neural network performance directly?

Word limit: **700**. Date: 2026-04-15. Points: 15.

---

## Submission answer (plain text)

**Q1.** I trained PyTorch MLP surrogates for all 8 capstone functions this week, comparing four regularisation variants per function: plain, dropout, weight decay, and a small 5-seed ensemble.

The regularisation choice was what mattered most. On the sparse-data functions like F1 and F3, a plain MLP overfit and couldn't beat the LOOCV baseline. Dropout and weight decay both helped, but they do different things (dropout adds noise during training, weight decay shrinks weights toward zero), and different functions ended up preferring different variants. Width (16 vs 32 hidden units) barely mattered once regularisation was in place. The ensemble was useful whenever no single variant was clearly winning, since it averages out seed-level noise. I kept the learning rate and early-stopping patience fixed across functions, which is a limitation I'd fix next time.

**Q2.** The continuous hyperparameters in my setup were the learning rate, the weight decay strength, and the dropout rate. These live on a smooth surface, so you can sensibly use Bayesian optimisation or a log-scale random search on them. The discrete ones were the number of layers, the width, the activation, the optimiser, the ensemble size, and the patience for early stopping. Those don't have a gradient, so you're stuck with grid search, random search, or something evolutionary.

Real setups always mix both, which is what makes tuning hard. In my pipeline I grid-searched a small discrete space of (variant, width) and just fixed the continuous values. Pragmatic, but the continuous side is under-tuned.

**Q3.** I can reuse the BBO tools I'm already using in the capstone (GP-based search, acquisition functions, cross-validation as the objective) to tune the NN's hyperparameters internally. The NN's learning rate, dropout, and weight decay are continuous inputs, and the CV RMSE on a given function is the response surface. It's the same approach I apply to the capstone functions, just scoped to a sub-problem inside the pipeline.

Worth flagging though: at 13–43 training points per function, the CV RMSE itself is pretty noisy, so aggressive hyperparameter search is as likely to overfit the metric as it is to find a genuinely better model. I'd keep the search space small rather than chase the last few percent.

---

## Submission answer (HTML version)

<p><strong>Q1.</strong> I trained PyTorch MLP surrogates for all 8 capstone functions this week, comparing four regularisation variants per function: plain, dropout, weight decay, and a small 5-seed ensemble.</p>

<p>The regularisation choice was what mattered most. On the sparse-data functions like F1 and F3, a plain MLP overfit and couldn't beat the LOOCV baseline. Dropout and weight decay both helped, but they do different things (dropout adds noise during training, weight decay shrinks weights toward zero), and different functions ended up preferring different variants. Width (16 vs 32 hidden units) barely mattered once regularisation was in place. The ensemble was useful whenever no single variant was clearly winning, since it averages out seed-level noise.</p>

<p><strong>Q2.</strong> The continuous hyperparameters in my setup were the learning rate, the weight decay strength, and the dropout rate. These live on a smooth surface, so you can sensibly use Bayesian optimisation or a log-scale random search on them. The discrete ones were the number of layers, the width, the activation, the optimiser, the ensemble size, and the patience for early stopping. Those don't have a gradient, so you're stuck with grid search or random search.</p>

<p>In my pipeline I grid-searched a small discrete space of (variant, width) and just fixed the continuous values. Pragmatic, but the continuous side is under-tuned.</p>

<p><strong>Q3.</strong> I can reuse the BBO tools I'm already using in the capstone (GP-based search, acquisition functions, cross-validation as the objective) to tune the NN's hyperparameters internally. The NN's learning rate, dropout, and weight decay are continuous inputs, and the CV RMSE on a given function is the response surface. It's the same approach I apply to the capstone functions, just scoped to a sub-problem inside the pipeline.</p>

<p>Worth flagging though: at 13–43 training points per function, the CV RMSE itself is pretty noisy, so aggressive hyperparameter search is as likely to overfit the metric as it is to find a genuinely better model. I'd keep the search space small rather than chase the last few percent.</p>
