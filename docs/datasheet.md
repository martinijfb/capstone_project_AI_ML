# Datasheet: BBO Capstone Data Set

## 1. Motivation

**Why was this data set created?**
To support the capstone challenge on black-box optimisation. The task is to find the maximum of eight unknown functions, where each evaluation is expensive and the only information available is the function output at a chosen input. The data set is the running log of every query made to the black-box oracle plus the value it returned.

**What task does the data support?**
Maximisation of eight black-box functions using surrogate modelling and informed search. It also supports model-selection discipline under a tight query budget.

## 2. Composition

**What does the data contain?**
For each of the eight functions there are two NumPy arrays stored in `data/function_N/`:
- `initial_inputs.npy`: an `(n, d)` array of query points where `d` is the function's input dimensionality.
- `initial_outputs.npy`: a `(n,)` array of the corresponding scalar outputs returned by the portal.

**Size and shape as of week 10:**

| Function | Dimensions | Total points | Initial | Weekly queries added |
|----------|------------|--------------|---------|----------------------|
| F1       | 2          | 19           | 10      | 9                    |
| F2       | 2          | 19           | 10      | 9                    |
| F3       | 3          | 24           | 15      | 9                    |
| F4       | 4          | 39           | 30      | 9                    |
| F5       | 4          | 29           | 20      | 9                    |
| F6       | 5          | 29           | 20      | 9                    |
| F7       | 6          | 39           | 30      | 9                    |
| F8       | 8          | 49           | 40      | 9                    |

**Format.** All inputs are floats in the unit hypercube `[0, 1)` with six decimal places. Outputs are 64-bit floats with no enforced range.

**Gaps and missing data.** None of the points are missing in the technical sense (every query has a recorded output).

**Sensitive content.** None. The data is synthetic outputs from anonymous numerical functions. No personal data, no protected attributes, no privacy concerns.

## 3. Collection process

**How was the data collected?**
- The initial points (10 to 40 per function) were provided by the course at the start of the project.
- Every week after that, I selected one input per function, formatted it according to the portal rules, submitted it through the capstone portal, and the portal returned a scalar output that I appended to the data.

**Sampling strategy.**
- Initial points: unknown to me.
- Weekly queries: not random. They are the output of a per-function decision pipeline that fits surrogate models (Ridge, KNN, Random Forest, SVR, Gradient Boosting, Gaussian Processes with several kernels, an MLP) using leave-one-out cross-validation, then selects a candidate either from the ensemble of model suggestions, from a trust-region method, or from a fall-back space-filling procedure when no model beats baseline.
**Time frame.** Weekly queries have been collected once a week from week 1 through week 10 (April to early June 2026).

**Ethical review.** Not applicable. No human subjects, no PII, no consent issues.

## 4. Pre-processing, cleaning and labelling

The raw `.npy` files are preserved exactly as received from the portal. No transformations are applied to the stored data.

Transformations are applied at analysis time only, inside the weekly notebooks.

## 5. Uses

**Intended uses.**
- Reproducing the per-week query selection by running the corresponding notebook top to bottom.
- Re-fitting any of the surrogate models documented in the model card on the same data.
- Inspecting trajectories, plots and decision markdowns from the weekly analysis.
- Academic discussion as part of the capstone learning outcome.

## 6. Distribution

**Where is it available?**
In this GitHub repository under `data/function_N/`. The repository is public so that peers and facilitators can review the work as required by the capstone instructions.

**Terms of use.**
The data, code and notebooks in this repository are personal academic work submitted as part of the programme. They are shared for educational discussion and peer review. There is no separate licence; treat it as the same status as any classroom submission.

**Costs or restrictions.**
No fees. No access restrictions beyond the GitHub repository being public.

## 7. Maintenance

**Maintainer.** Martin Fernandes.

**Updates.** The data set grows by one row per function each week through week 12, after which the capstone ends and no further updates are expected.

**Versioning.** Git history records every weekly append. The point counts in this datasheet are correct as of week 10.

**Archival.** After the capstone ends the repository will remain public for reference. No long-term maintenance is planned, since the underlying portal that produced the outputs will no longer be active.
