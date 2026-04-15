# Week 04 Reflection — Neural Networks & Decision Boundaries

## Portal prompts (7)

1. Which inputs acted like support vectors — points near a decision boundary or region of rapid change?
2. Did you train a surrogate NN and explore input→output gradients?
3. How would you frame BBO as a classification task? Trade-offs between misclassification and exploration?
4. Which model type felt most appropriate — linear regression, SVM, or NN? How did you balance interpretability vs flexibility?
5. Which input variables showed the steepest gradients in your NN?
6. How effectively did the NN approximate the decision boundary vs an SVM? Did backpropagation help visualise it?
7. Did the NN capture non-linear patterns better than simpler models? Was the added complexity worth it?

Word limit: **700**. Date: 2026-04-15.

---

## Submission answer (plain text)



**Q1 — Support vectors.** Two patterns emerged. On F4, the current best is clearly a support vector, a small perturbation in W2 caused a huge drop in Y. I responded by explicitly shrinking the step size of my RMSE-weighted ensemble. On F1, where almost all observations are numerically zero, two non-zero points carry the only real magnitude and alone define the sign boundary. I placed the next query equidistant from both to maximise classifier information.

**Q2 — NN surrogates and gradients.** I trained PyTorch MLPs across all 8 functions, comparing regularisation variants (plain / dropout / weight-decay / small ensembles) via K-fold CV. `torch.autograd.backward` gave me dY/dx at each current-best, which I used as a directional hint combined with ensemble agreement so I don't jump off the data manifold.

**Q3 — Classification framing and trade-offs.** F1 is the function where I use this framing. An RBF-kernel SVM captures its non-linear sign boundary cleanly, whereas logistic regression can't beat the majority-class baseline. The trade-off is between conservatism and coverage: a narrow decision boundary keeps queries safe inside the known positive region but misses peaks just outside; a wide one wastes queries in negative territory. I lean conservative on F1 because the classifier's training data is still sparse — I'd rather gather more before fully trusting it.

**Q4 — Model choice.** I don't privilege any family; each competes per function on LOOCV RMSE. Linear models (Ridge) are almost always excluded as boundary extrapolators. SVR led on F4/F6 in earlier weeks but hit boundaries on F2/F6/F8 this week, with GP-Matern taking over. NN is always included alongside but never topped any leaderboard. I prefer the interpretable ensemble over a single NN's argmax because the per-dim convergence-spread tells me which dimensions to trust.

**Q5 — Steepest gradients.** On F4 the NN gradient agreed with my ensemble direction and flagged that two dimensions dominate while a third is nearly flat. On F2 the gradient corroborated the ensemble's push on x2.

**Q6 — NN classifier vs SVM.** For F1's sign task I trained both: an RBF-SVM and a small NN classifier. The SVM won comfortably. At small sample sizes a kernel SVM has a closed-form margin with explicit regularisation, while the NN has many weights to fit with nothing protecting it from overfitting — the NN underperformed the majority-class baseline. Backprop could have given me gradient maps through the NN's predicted probability surface, but at its accuracy they'd mostly encode noise. The SVM's decision function gave me a cleaner visualisation, so that's what I used.

**Q7 — Was NN complexity worth it?** NNs beat the LOOCV baseline on most functions but never ranked first. Their real value this week was gradient analysis, not predictive accuracy. On F5 and F8, tuned GPs matched or exceeded the NN with less tuning overhead. At current sample sizes, GPs and SVRs remain my primary models; NNs contribute a diversifying vote and interpretable gradients.



---

## Submission answer (HTML version)

<p><strong>Q1</strong> Two patterns emerged. On F4, the current best is clearly a support vector, a small perturbation in W2 caused a huge drop in Y. I responded by explicitly shrinking the step size of my RMSE-weighted ensemble. On F1, where almost all observations are numerically zero, two non-zero points carry the only real magnitude and alone define the sign boundary. I placed the next query equidistant from both to maximise classifier information.</p>

<p><strong>Q2</strong> I trained PyTorch MLPs across all 8 functions, comparing regularisation variants (plain / dropout / weight-decay / small ensembles) via K-fold CV. <code>torch.autograd.backward</code> gave me dY/dx at each current-best, which I used as a directional hint combined with ensemble agreement so I don't jump off the data manifold.</p>

<p><strong>Q3</strong> F1 is the function where I use this framing. An RBF-kernel SVM captures its non-linear sign boundary cleanly, whereas logistic regression can't beat the majority-class baseline. The trade-off is between conservatism and coverage: a narrow decision boundary keeps queries safe inside the known positive region but misses peaks just outside; a wide one wastes queries in negative territory. I lean conservative on F1 because the classifier's training data is still sparse. I'd rather gather more before fully trusting it.</p>

<p><strong>Q4</strong> I don't privilege any family. Each competes per function on LOOCV RMSE. Linear models (Ridge) are almost always excluded as boundary extrapolators. SVR led on F4/F6 in earlier weeks but hit boundaries on F2/F6/F8 this week, with GP-Matern taking over. NN is always included alongside but never topped any leaderboard. I prefer the interpretable ensemble over a single NN's argmax because the per-dim convergence-spread tells me which dimensions to trust.</p>

<p><strong>Q5</strong> On F4 the NN gradient agreed with my ensemble direction and flagged that two dimensions dominate while a third is nearly flat. On F2 the gradient corroborated the ensemble's push on x2.</p>

<p><strong>Q6</strong> For F1's sign task I trained both: an RBF-SVM and a small NN classifier. The SVM won comfortably. At small sample sizes a kernel SVM has a closed-form margin with explicit regularisation, while the NN has many weights to fit with nothing protecting it from overfitting. The NN underperformed the majority-class baseline. Backprop could have given me gradient maps through the NN's predicted probability surface, but at its accuracy they'd mostly encode noise. The SVM's decision function gave me a cleaner visualisation, so that's what I used.</p>

<p><strong>Q7</strong> NNs beat the LOOCV baseline on most functions but never ranked first. Their real value this week was gradient analysis, not predictive accuracy. On F5 and F8, tuned GPs matched or exceeded the NN with less tuning overhead. At current sample sizes, GPs and SVRs remain my primary models; NNs contribute a diversifying vote and interpretable gradients.</p>
