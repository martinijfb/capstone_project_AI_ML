# Pipeline Upgrade Research: TuRBO, BoTorch, NeurIPS 2020 BBO Challenge

**Date compiled:** 2026-05-03 (end of W6)
**Purpose:** evaluate techniques from the published BO literature for inclusion in W7+ queries.
**Constraints to remember:** one query per function per week (q=1, sequential), 6 weeks remaining, 8 functions of varying dimensionality, current pipeline already uses sklearn ensemble + GP-Matern at ν ∈ {0.5, 1.5, 2.5} + RBF + PyTorch NN with autograd gradient extraction.

---

## TuRBO (Eriksson et al., NeurIPS 2019)

**Core idea.** Instead of fitting one global GP and chasing a global acquisition function (which over-explores in higher dimensions), TuRBO maintains one or more *trust regions* (TRs), each with its own *local GP*, and only proposes points inside the active TR. The local GP gives sharper, less-misled posteriors because it only fits nearby data; the TR machinery acts as a learned exploration radius.

**Trust region geometry.** Each TR is a hyperrectangle centered on the current best observation (in the noise-free case). The side length in dimension d is `L * lengthscale_d / (geometric mean of all lengthscales)`, so the GP's own ARD lengthscales stretch the box, narrowing it on dimensions the GP thinks matter and widening it on inert ones. `L` is the single scalar that controls the box's overall scale.

**L adaptation (the contract/expand logic):**
- **Initialization:** `L_init = 0.8` (in unit cube).
- **Bounds:** `L_max = 1.6`, `L_min = 0.5^7 ≈ 0.0078`.
- **Success counter** increments when a new point improves the best-so-far by more than `1e-3 * |best|`. **Failure counter** increments on no improvement.
- **After tau_succ = 3 consecutive successes:** `L <- min(2L, L_max)` (expand).
- **After tau_fail consecutive failures (≈ ceil(max(4/q, dim/q)) where q = batch size):** `L <- L/2` (contract).
- When `L < L_min`, the TR is killed and a new one is restarted from a fresh random Latin hypercube.

**Local vs global GP.** The local GP is a vanilla Matern-5/2 GP with ARD; it is just *fit only on data inside the current TR* (or a recent window). This is what makes TuRBO scale to D = 100+ where global GPs collapse: the local model doesn't have to explain distant, structurally different parts of the landscape.

**Batch vs sequential.** TuRBO-1 (single TR) is the natural sequential variant. Within a batch it uses Thompson sampling across many candidates in the TR; with batch=1 (our case) it just picks the single best TS draw. Multiple TRs (TuRBO-m) need batch evaluations to keep all TRs progressing. **TuRBO-1 with q=1 is exactly the regime our one-query-per-week constraint requires.** Reference implementation: [uber-research/TuRBO](https://github.com/uber-research/TuRBO); also packaged as a tutorial in BoTorch.

---

## BoTorch

**What it adds over scikit-learn GP.** sklearn's `GaussianProcessRegressor` is a numpy/scipy implementation with no GPU support, no autograd, and basically no acquisition-function machinery (you have to hand-roll EI/UCB). BoTorch is built on PyTorch + GPyTorch and gives:
1. **autograd through the acquisition function**, so optimization uses true gradients instead of finite differences or evolutionary search;
2. **Monte Carlo acquisition** (qEI, qUCB, qNEI) that handles batch and noisy observations;
3. **input/outcome transforms** (Normalize, Standardize) wired into the model so we don't have to renormalize manually each round;
4. **fully Bayesian GPs** via NUTS/HMC for the SAAS prior;
5. **proper handling of pending observations** for async settings.

**Acquisition functions** ([botorch.org/docs/acquisition](https://botorch.org/docs/acquisition/)):
- **UCB / qUCB:** `mu(x) + sqrt(beta) * sigma(x)`. Cheap, smooth, easy to optimize. The "kappa scheduling" the peer is doing on F5 likely means starting beta high (explore) and decaying it (exploit). Classical Srinivas et al. theory says `beta_t = 2 log(t^2 * pi^2 / (6 * delta))` but in practice everyone uses a hand-tuned schedule like `beta = 2.0 -> 0.5` linearly.
- **EI / qEI / qNEI (Noisy EI):** Expected improvement over current best. qNEI is the modern default: robust to noise, integrates over uncertainty about which point is best.
- **KG (Knowledge Gradient):** Look-ahead acquisition that asks "if I observed at x, how much would the posterior max move?" Much more expensive (~ inner BO loop per evaluation) but theoretically optimal for one-shot regret. BoTorch's "one-shot" SAA formulation is a key contribution of the paper.
- **MES (Max-value Entropy Search):** Information-theoretic, picks point that maximally reduces entropy of `argmax`. Strong on multi-modal landscapes.
- **PI:** Probability of improvement, included but generally dominated by EI.

**TuRBO in BoTorch.** Yes, there is an official tutorial ([turbo_1](https://botorch.org/docs/tutorials/turbo_1/)) showing the full state machine implemented in ~80 lines on top of `SingleTaskGP` + Thompson sampling. It's batch-by-default but trivially configurable to `BATCH_SIZE = 1`.

**Integration with existing PyTorch surrogates.** Our existing NN-with-autograd workflow is already PyTorch-native, so BoTorch slots in as a parallel model family alongside the sklearn ensemble. We'd build a `SingleTaskGP(X, Y, input_transform=Normalize(d), outcome_transform=Standardize(1))`, fit with `fit_gpytorch_mll`, then call `optimize_acqf(qLogNoisyEI(model, X), bounds, q=1, num_restarts=10, raw_samples=512)`.

**Ax (Meta's higher-level layer).** Wraps BoTorch with experiment management, automatic model selection, generation strategies, and built-in support for SAASBO ([SAAS](https://botorch.org/docs/tutorials/saasbo/)): a sparse-prior fully Bayesian GP designed for high-D problems where only a few dimensions matter. Probably overkill for our 8-function setup, but **SAASBO is highly relevant for F8 (8D)** if we suspect only 3-4 dimensions actually drive the output.

---

## NeurIPS 2020 BBO Challenge

**Problem.** Tune scikit-learn / xgboost / MLP hyperparameters on UCI datasets to minimize CV loss. The benchmark harness is **Bayesmark** ([rdturnermtl/bbo_challenge_starter_kit](https://github.com/rdturnermtl/bbo_challenge_starter_kit)). Each problem allowed 16 batches × 8 suggestions = 128 evaluations. Scoring: 0 = same as one random sample, 100 = global optimum every time.

**Winners:**
1. **HEBO (Huawei Noah's Ark)** - 1st place. Key ideas (paper [arxiv 2012.03826](https://arxiv.org/abs/2012.03826)):
   - **Input warping** (Kumaraswamy CDF) and **output warping** (Box-Cox / Yeo-Johnson) to fight non-stationarity and skewed Y distributions.
   - **Heteroscedastic noise modeling**: variance learned per region, not assumed constant.
   - **Multi-objective acquisition ensemble**: optimize EI + PI + UCB simultaneously and pick from the Pareto front. Hedges against any one acquisition's failure mode.
   - **Robust acquisition maximizers** (multiple restarts + careful initialization).
2. **JetBrains** - 3rd place. "Learning Search Space Partition for Local Bayesian Optimization" ([repo](https://github.com/jbr-ai-labs/bbo-challenge-jetbrains-research)). Builds on TuRBO: adds a learned space-partitioning layer that decides where to place the trust region based on past data, rather than always centering on `argmax`.
3. **Optuna Developers** - TPE (Tree-structured Parzen Estimator) baseline with careful tuning, scored ~92.

**Documented lessons** (Turner et al. 2021 post-mortem, [PMLR v133](https://proceedings.mlr.press/v133/turner21a.html)):
- BO **decisively beats random search**. Average winning score ~92 vs random's 0; the gap is largest in the first 30-50 evaluations (sample-efficiency regime).
- **Output warping was the single largest gain.** Multiple top-5 entries adopted it. Y distributions in real problems are wildly skewed and breaking that with a learned warp helped GPs enormously.
- **Ensembles of acquisition functions** outperformed any single acquisition. The "best" acquisition varies by problem.
- **Local methods (TuRBO + variants) dominated higher-D problems**; pure global GPs underperformed past ~6D.
- Random restarts of the optimizer of the acquisition function matter a lot. Many gains came from better acquisition optimization, not better surrogates.

**What transfers to our setting:**
- Output warping, acquisition ensembling, TuRBO trust regions: all directly applicable.
- HEBO's heteroscedastic noise is less critical for us (deterministic black box).
- The batch-of-8 evaluation regime doesn't transfer (we're sequential), but every winner's underlying surrogate+acquisition logic does.

---

## Synthesis: What to apply in the next 6 weeks

We're sequential, q=1, with 6 weeks left across 8 functions. The peer's GP-UCB result on F5 shows a properly fit GP can crush a heuristic ensemble. Prioritized by impact-per-effort:

### 1. TuRBO-1 with q=1, drop-in for our weakest functions [HIGH PRIORITY, MEDIUM EFFORT]

**Why:** TuRBO-1 with batch=1 is exactly the sequential regime we need. The trust-region contract/expand state is just three integers (`L`, `succ_count`, `fail_count`) we persist in a JSON file across weeks. No batch required.

**Where:** **F5 (4D, we're behind), F6 (5D), F7 (6D), F8 (8D)**. All the higher-dim functions where our global ensemble is likely over-exploring. F5 in particular: the peer's success suggests the landscape rewards local exploitation around a strong basin.

**Effort:** ~1 day. Copy [BoTorch turbo_1 tutorial](https://botorch.org/docs/tutorials/turbo_1/), set `BATCH_SIZE=1`, persist `(L, succ, fail, best_x)` in `data/function_N/turbo_state.json`. Each week: load state, fit local GP on data inside the TR, draw Thompson samples, pick the max, submit.

**Risk:** TR can collapse onto a local optimum on multi-modal landscapes. **Mitigation:** keep the existing ensemble query as a "second opinion" and only switch to TuRBO if it agrees, OR run TuRBO on 5/6/7/8 only and keep the current pipeline on 1/2/3/4 where it's working.

### 2. GP-UCB with proper kappa schedule (BoTorch SingleTaskGP + qUCB) [HIGH PRIORITY, LOW EFFORT]

**Why:** The peer has empirical evidence this works on F5. Our current GP code uses sklearn-flavored Matern; a BoTorch `SingleTaskGP` with `Normalize`/`Standardize` transforms and ARD Matern-5/2 fit by `fit_gpytorch_mll` will be measurably better-calibrated. Then `qUpperConfidenceBound(model, beta=beta_t)` with `beta_t = 2.0 * (0.5)^(week/12)`: start exploratory, exponentially decay to exploitative by week 12.

**Where:** F5 (immediate test against the peer's score), F2 (2D, GP should be near-optimal), F4.

**Effort:** ~3 hours. ~30 lines on top of existing data loaders.

**Risk:** None really. Run it as one of the candidate generators and gate it through the existing baseline-beating check. If it regresses on F2 we just don't submit it.

### 3. Output warping (Yeo-Johnson on Y) for all surrogates [MEDIUM PRIORITY, LOW EFFORT]

**Why:** The biggest documented lesson from BBO 2020. F1 has the well-known sign-flipping issue, that's a textbook case where Y-warping helps. Apply `sklearn.preprocessing.PowerTransformer(method='yeo-johnson')` to Y before fitting any model, then inverse-transform predictions.

**Where:** F1 (skewed Y, two large negatives), F3 (if Y distribution is skewed), and as a default for any function where the top-5 / median Y ratio is > 5x.

**Effort:** ~2 hours. One wrapper around existing model fits.

**Risk:** Yeo-Johnson can be unstable with very few points (<10). Keep the current F1 classifier+log-SVR pipeline as primary; use warped models as a third opinion.

### 4. Acquisition ensemble (EI + UCB + qNEI vote) instead of RMSE-weighted argmax [MEDIUM PRIORITY, MEDIUM EFFORT]

**Why:** HEBO's documented secret sauce. Our current "RMSE-weighted argmax across model predictions" picks the point with highest predicted Y, which is essentially a greedy mean strategy with no exploration term. Swapping in 3 different acquisition functions over the same GP and selecting via Pareto-front voting (or just majority vote) directly transfers HEBO's approach.

**Where:** F4 (SVR dominates at 63%, adding acquisition diversity could break out of local exploitation), F6 (moderate model, needs exploration).

**Effort:** ~1 day. Need to wrap sklearn models in a uniform `predict_mean_var` interface so EI/UCB/PI can be computed against any of them.

**Risk:** Could actively regress on functions where greedy exploitation is currently winning (F4). **Mitigation:** A/B against current pipeline; only switch on functions where current trajectory is flat for 2+ weeks.

### 5. SAASBO via Ax for F8 only [LOWER PRIORITY, MEDIUM EFFORT]

**Why:** F8 is 8D and we have 45 points. Standard GPs overfit ARD lengthscales catastrophically here. SAASBO's sparse Hamiltonian-MC prior is *purpose-built* for "high-D where only a few dimensions matter", a plausible structure for F8 (RF importance shows x1 + x3 dominate at ~0.4 each, others <0.1).

**Where:** F8 only (do not generalize, overkill for lower-D).

**Effort:** ~1 day. Ax wrapper [SAASBO tutorial](https://botorch.org/docs/tutorials/saasbo/) is paste-and-go; main cost is HMC fitting time (~1 minute per query, irrelevant at 1/week).

**Risk:** With 45 points in 8D, SAASBO has just enough signal to work but isn't guaranteed to extract more than a well-tuned ARD GP. Worth one query-week experiment in week 7 or 8. The inverse-lengthscale posteriors are themselves valuable diagnostics for the reflection regardless.

---

## What to skip

- **Full HEBO library install:** too heavyweight for one query/week, and most of HEBO's gains come from input/output warping + acquisition ensemble, which we can implement à la carte.
- **Batch acquisition (qEI with q>1):** our constraint is hard q=1; no benefit.
- **Knowledge Gradient:** implementation cost is high, theoretical benefit assumes large eval budget; not worth it for 6 weeks.
- **CMA-ES / evolutionary methods:** these need population evals per iteration, fundamentally incompatible with q=1.

---

## Suggested W7 first steps

1. **Friday or weekend before W7 query:** install BoTorch into the uv env (`uv add botorch ax-platform`).
2. **Implement TuRBO state file** for F5 with initial L=0.8, succ=fail=0, best_x=current best.
3. **Add one notebook cell per upgrade** to `notebooks/week_07.ipynb`, running them as alternative candidate generators alongside the existing pipeline.
4. **Decision rule for W7:** for each function, submit the existing pipeline's query UNLESS the new method (TuRBO or qUCB or warped-Y ensemble) produces a candidate that (a) passes the same boundary/outlier filters AND (b) the underlying model has a measurably better LOOCV RMSE than the current best model. This way the new methods are gated by the same validate-then-trust principle.
5. **Document outcomes** in a new file `suggestions/pipeline_upgrade_evaluation.md` after W7 results land.

---

## Sources

- TuRBO paper: [arxiv.org/abs/1910.01739](https://arxiv.org/pdf/1910.01739) and [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2019/hash/6c990b7aca7bc7058f5e98ea909e924b-Abstract.html)
- TuRBO reference impl: [github.com/uber-research/TuRBO](https://github.com/uber-research/TuRBO)
- TuRBO in BoTorch: [botorch.org/docs/tutorials/turbo_1](https://botorch.org/docs/tutorials/turbo_1/)
- BoTorch paper: [NeurIPS 2020 proceedings](https://proceedings.neurips.cc/paper/2020/file/f5b1b89d98b7286673128a5fb112cb9a-Paper.pdf)
- BoTorch acquisitions: [botorch.org/docs/acquisition](https://botorch.org/docs/acquisition/)
- Ax (Meta): [github.com/facebook/Ax](https://github.com/facebook/Ax)
- SAASBO tutorial: [botorch.org/docs/tutorials/saasbo](https://botorch.org/docs/tutorials/saasbo/)
- HEBO paper: [arxiv.org/abs/2012.03826](https://arxiv.org/abs/2012.03826) and [github.com/huawei-noah/HEBO](https://github.com/huawei-noah/HEBO)
- JetBrains entry: [github.com/jbr-ai-labs/bbo-challenge-jetbrains-research](https://github.com/jbr-ai-labs/bbo-challenge-jetbrains-research)
- BBO Challenge post-mortem (Turner et al. 2021): [PMLR v133](https://proceedings.mlr.press/v133/turner21a.html)
- BBO Challenge starter kit + Bayesmark: [github.com/rdturnermtl/bbo_challenge_starter_kit](https://github.com/rdturnermtl/bbo_challenge_starter_kit)
