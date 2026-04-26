# Week 05 — Capstone 16.2 Reflection: Software Architecture

## Portal prompts (3 sections)

**Repository structure**
1. How have you organised your repository so far (e.g. data, notebooks, queries, results)?
2. What changes will you make to improve clarity, navigability and reproducibility?

**Coding libraries and packages**
3. Which libraries or frameworks (e.g. PyTorch, TensorFlow, scikit-learn) are central to your approach?
4. Why are these choices appropriate for your problem, and what trade-offs did you consider?

**Documentation**
5. How do your README and other documents currently describe the purpose, inputs, outputs and objectives of your BBO capstone project?
6. What updates do you need to align the documentation with your most recent strategy and results?

Word limit: **700**. Date: 2026-04-22.

---

## Submission answer (plain text)

**Repository structure.** The repo is organised by week and by purpose, parallel to each other. Top-level folders: data/, notebooks/, plots/, models/, weekly_queries/, reflections/, suggestions/, src/, and a docs/ folder where I keep my pipeline notes (the cell sequence, the decision tree, the per-week procedure) so I can refer back to them at the start of each round. Within plots/ and models/, things that change every week live in per-week subfolders so old weeks don't get clobbered. Reusable code sits in src/ (a small nn_models.py module for the PyTorch surrogates, an add_results.py script for appending portal responses, and a few utility helpers). The README is the public face of the project.

What I'd change. Honestly, not much. The parallel-by-purpose layout has held up across five rounds without any reorganisation, and the per-week subfolders mean I can revisit any earlier round cleanly. The one improvement I'd consider for completeness is consolidating per-function decision histories. Right now the trajectory of any single function across five weeks lives in five separate notebook files, and you'd have to read all of them to reconstruct it. A small summary file per function in docs/ would solve that without changing the rest of the structure.

**Libraries.** scikit-learn does most of the heavy lifting: Ridge, KNN, Random Forest, SVR, Gradient Boosting, Gaussian Process, all wrapped in GridSearchCV with LeaveOneOut for the small datasets I have. PyTorch is used only for the NN surrogates, mostly because of the autograd API that lets me extract dY/dx at any point cheaply. scipy.spatial.distance.cdist powers the balanced Voronoi space-filling. matplotlib and numpy are the rest.

Why these choices. sklearn fits the problem because the data is tiny (between 13 and 44 points per function depending on the round) and CPU-based, so I don't need anything distributed. Cross-validation and hyperparameter grids are first-class in sklearn. PyTorch over TensorFlow because I care about gradient extraction at runtime, not training scale, and PyTorch's autograd makes that natural.

Trade-offs. I considered BoTorch for the Gaussian Process pieces because it has built-in acquisition functions for Bayesian optimisation, but sklearn's GP is good enough at this data scale and adds no new dependency. I tried XGBoost briefly but the speed advantage doesn't matter at this sample size, so I dropped it to keep the dependency list small. TensorFlow would have been equivalent for the surrogate models but it would have meant rewriting the gradient extraction code.

**Documentation.** The README has four sections: project overview, the input/output format, challenge constraints, and a technical-approach summary. I refresh the technical-approach section every week so it stays current with the latest model families and decision rules. Beyond the README, each week produces three durable artifacts: weekly_queries/week_XX.md (what I submitted and why), reflections/week_XX.md (what I learned), and suggestions/suggestions_for_week_(XX+1).md (conditional strategies for next week based on what the portal might return). Together these let me reconstruct any week's reasoning without rereading the whole notebook.

Updates needed. Section 4 of the README is the most likely thing to drift. I refresh it weekly but the structure could use a small refactor to separate "consistent across all functions" rules from "function-specific" exceptions like F1's classifier-and-log-SVR or F3's outlier-aware analysis. A top-level docs/ folder with per-function decision histories would also help anyone trying to follow what happened on a specific function over five weeks. Right now that information is scattered across five notebook files and you'd have to read all of them to reconstruct a single function's trajectory.

---

## Submission answer (HTML version)

<p><strong>Repository structure.</strong> The repo is organised by week and by purpose, parallel to each other. Top-level folders: <code>data/</code>, <code>notebooks/</code>, <code>plots/</code>, <code>models/</code>, <code>weekly_queries/</code>, <code>reflections/</code>, <code>suggestions/</code>, <code>src/</code>, and a <code>docs/</code> folder where I keep my pipeline notes (the cell sequence, the decision tree, the per-week procedure) so I can refer back to them at the start of each round. Within <code>plots/</code> and <code>models/</code>, things that change every week live in per-week subfolders so old weeks don't get clobbered. Reusable code sits in <code>src/</code> (a small <code>nn_models.py</code> module for the PyTorch surrogates, an <code>add_results.py</code> script for appending portal responses, and a few utility helpers). The README is the public face of the project.</p>

<p>What I'd change. Honestly, not much. The parallel-by-purpose layout has held up across five rounds without any reorganisation, and the per-week subfolders mean I can revisit any earlier round cleanly. The one improvement I'd consider for completeness is consolidating per-function decision histories. Right now the trajectory of any single function across five weeks lives in five separate notebook files, and you'd have to read all of them to reconstruct it. A small summary file per function in <code>docs/</code> would solve that without changing the rest of the structure.</p>

<p><strong>Libraries.</strong> scikit-learn does most of the heavy lifting: Ridge, KNN, Random Forest, SVR, Gradient Boosting, Gaussian Process, all wrapped in GridSearchCV with LeaveOneOut for the small datasets I have. PyTorch is used only for the NN surrogates, mostly because of the autograd API that lets me extract dY/dx at any point cheaply. <code>scipy.spatial.distance.cdist</code> powers the balanced Voronoi space-filling. matplotlib and numpy are the rest.</p>

<p>Why these choices. sklearn fits the problem because the data is tiny (between 13 and 44 points per function depending on the round) and CPU-based, so I don't need anything distributed. Cross-validation and hyperparameter grids are first-class in sklearn. PyTorch over TensorFlow because I care about gradient extraction at runtime, not training scale, and PyTorch's autograd makes that natural.</p>

<p>Trade-offs. I considered BoTorch for the Gaussian Process pieces because it has built-in acquisition functions for Bayesian optimisation, but sklearn's GP is good enough at this data scale and adds no new dependency. I tried XGBoost briefly but the speed advantage doesn't matter at this sample size, so I dropped it to keep the dependency list small. TensorFlow would have been equivalent for the surrogate models but it would have meant rewriting the gradient extraction code.</p>

<p><strong>Documentation.</strong> The README has four sections: project overview, the input/output format, challenge constraints, and a technical-approach summary. I refresh the technical-approach section every week so it stays current with the latest model families and decision rules. Beyond the README, each week produces three durable artifacts: <code>weekly_queries/week_XX.md</code> (what I submitted and why), <code>reflections/week_XX.md</code> (what I learned), and <code>suggestions/suggestions_for_week_(XX+1).md</code> (conditional strategies for next week based on what the portal might return). Together these let me reconstruct any week's reasoning without rereading the whole notebook.</p>

<p>Updates needed. Section 4 of the README is the most likely thing to drift. I refresh it weekly but the structure could use a small refactor to separate "consistent across all functions" rules from "function-specific" exceptions like F1's classifier-and-log-SVR or F3's outlier-aware analysis. A top-level <code>docs/</code> folder with per-function decision histories would also help anyone trying to follow what happened on a specific function over five weeks. Right now that information is scattered across five notebook files and you'd have to read all of them to reconstruct a single function's trajectory.</p>
