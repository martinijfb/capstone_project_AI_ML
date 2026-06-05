# Model Card: BBO Capstone Optimisation

## Model overview

Name: Validate-then-trust BBO pipeline. Version: v0.10 (week 10 of 12). Type: surrogate-model-driven optimisation; multi-family regression with leave-one-out cross-validation, plus trust-region Bayesian optimisation for stalled functions. Deployment: weekly query selection for an academic black-box optimisation challenge.

## Intended use

Selecting one query per week for each of eight black-box functions in the capstone. Target users: me, peers and facilitators reviewing the work.

## Training data

The growing data set described in [datasheet.md](datasheet.md). For each function: a small set of initial random points (10 to 40) plus one new point per week. Total sizes range from 19 (F1, F2) to 49 (F8) as of week 10. No transformations are written to disk; all pre-processing happens in-notebook.

## Inputs / outputs

Input: a vector in `[0, 1)^d` with `d` between 2 and 8. Output: a single query coordinate per function per week, formatted with six decimal places, hyphen-separated.

## Methods

Each function is fit with a panel of models (Ridge, KNN, Random Forest, SVR, Gradient Boosting, Gaussian Processes with Matern and RBF kernels, an MLP). Models that beat the standard-deviation baseline under leave-one-out cross-validation are pooled, and their argmax candidates are checked for boundary artifacts. The week's query is taken from either: an ensemble of interior model suggestions, a per-dimension hybrid (ensemble where models agree, top-K centroid where they disagree), or a trust-region Bayesian method (TuRBO with multi-kernel Thompson sampling) when standard refinement stalls.

## Performance

Metric: leave-one-out RMSE relative to the standard-deviation baseline for model selection; running best Y on the portal for outcome tracking.

## Assumptions and limitations

Assumes local smoothness in each function (so that Gaussian Process and SVR kernels with Matern or RBF can generalise from nearby points), and that leave-one-out cross-validation on a couple of dozen points is a meaningful proxy for true model quality. Limitations: a tight sample budget (one query per function per week).

## Ethical considerations

Sandbox academic project. No human subjects, no protected attributes, no real-world deployment. Transparency: every query is computed from data and saved models inside the notebook.

## Distribution

Model card and code available in this public GitHub repository alongside the data set, notebooks, weekly reflections and the decision framework. No separate licence; same status as the rest of the capstone submission.
