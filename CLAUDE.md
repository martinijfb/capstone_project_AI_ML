# Capstone Project: Black-Box Optimisation (BBO)

## Context

This is a capstone project for the Imperial College ML/AI programme (Modules 12–25). The goal is to find the maximum of 8 unknown black-box functions. Any ML technique can be used — the project spans the entire programme and you're encouraged to apply methods as you learn them.

## Rules

- **One query per function per week** — submitted through the Emeritus capstone portal
- **All functions are maximisation problems**
- Functions increase in dimensionality: F1-F2 (2D), F3 (3D), F4-F5 (4D), F6 (5D), F7 (6D), F8 (8D)
- Perfect solutions are NOT expected — the grading values thoughtful process, iteration, and reflection
- Input format: `x1-x2-x3-...-xn` — each value 0.000000 to 0.999999, exactly 6 decimal places, hyphen-separated, no spaces
- Reflection must be posted on the discussion board (under 700 words)

## Project Structure

```
capstone_project_AI_ML/
├── data/                     # raw data, grows each week
│   ├── function_1/
│   │   ├── initial_inputs.npy
│   │   └── initial_outputs.npy
│   └── ...
├── notebooks/                # one notebook per week, covers all 8 functions
│   ├── week_01.ipynb
│   └── ...
├── src/                      # reusable code across weeks
│   ├── utils.py
│   ├── optimizers.py
│   └── nn_models.py          # PyTorch MLP surrogates (Week 04+)
├── models/                   # saved NN weights + metadata (Week 04+)
│   ├── week_04/
│   │   ├── function_1_nn.pt
│   │   └── ...
│   └── week_XX/              # subfolder per week — models don't clash across weeks

├── weekly_queries/           # formatted submissions per week
│   ├── week_01.md
│   └── ...
├── reflections/              # weekly write-ups
│   ├── week_01.md
│   └── ...
├── suggestions/              # per-week recommendations for the next week
│   ├── suggestions_for_week_02.md
│   └── ...
└── plots/                    # saved figures organized by week
    ├── week_01/
    └── ...
```

## Weekly Pipeline

### Phase 1: Setup
1. `/new-week XX` — create notebook and folders
2. `/add-results` — append last week's portal response (if not already done)
3. `/status` — see current state of all 8 functions, track improvements
4. *(Week 04+)* `/train-nns` — train NN surrogates for all 8 functions (saves to `models/`)

### Phase 2: Analysis (per function)
5. `/analyze N` — full analysis pipeline:
   - Visualise: parallel coords, correlations, top vs bottom boxplots
   - Feature importance robustness (with/without best point)
   - Model grid search with LOOCV RMSE (compare against baseline)
   - Model convergence analysis per dimension (STRONG/moderate/weak)
   - Apply decision framework to choose query

### Phase 3: Submission
4. `/submit` — validate format + save to `weekly_queries/week_XX.md`
5. `/reflect` — draft reflection answering the portal questions, save to `reflections/week_XX.md`
6. `/update-readme` — update Section 4 of README.md with current week's strategies
7. Submit queries on portal + post reflection on discussion board

### Phase 4: Results (next week)
7. `/add-results` — append new data, compare predicted vs actual Y
8. Review `suggestions/suggestions_for_week_XX.md` for pre-planned strategies

## Decision Framework (developed in Week 01, extended W03-W04)

For each function, follow this process:

1. **If no models beat baseline** (e.g. F3 with 15pts/3D): use Y-weighted centroid of top 4. If top-K is clustered in one region, use **balanced Voronoi** instead.
2. **If one model dominates** (e.g. F4 SVR at 63%): trust that model if suggestion is interior
3. **If models are moderate** (e.g. F6 SVR at 30%): use best model, verify against centroid
4. **If models are weak but have consensus on some dims** (e.g. F7): centroid + override only STRONG consensus dimensions
5. **If models are strong but disagree on some dims** (e.g. F8): trust best model on dimensions with strong correlation + robust importance + consensus; centroid on the rest

**F1 special case**: F1 has tiny positive magnitudes and two large negatives — no raw regressor beats baseline. Always run the **classifier + log-SVR combined** approach first (see `suggestions/f1_long_term_strategy.md`). Fall back to **balanced Voronoi** only if the combined candidate sits at the sign boundary or log-SVR is miscalibrated.

**NN surrogates (Week 04+)**: `/train-nns` pre-trains an MLP surrogate per function and saves to `models/week_XX/function_N_nn.pt`. In Step 3 of `/analyze`, load via `nn_models.load_nn(N)` and treat as an additional model family — same baseline-beating gate, same boundary filtering, same convergence check as sklearn models. Extract `meta['gradient_at_best']` for reflection Q5 (steepest-gradient dimensions).

**Always check:**
- Does the suggestion hit a boundary? (any dim < 0.02 or > 0.98) → reject
- Is the feature importance robust? (re-run RF without best point — if importance drops >50%, it was inflated by one outlier)
- Do multiple model configs agree? (convergence spread < 0.2 = strong, < 0.4 = moderate, > 0.4 = weak)
- Is space-filling a corner? If using Voronoi, always use the **balanced** variant that penalizes boundary-proximity equally with cluster-proximity

## Techniques Available (by Programme Module)

### Already covered (Modules 1–14)
- **Probability & statistics**: Monte Carlo simulations, MLE, bootstrapping, distributions
- **Regression**: linear regression, correlation, feature engineering
- **Evaluation**: confusion matrix, precision/recall, F1, RMSE, k-fold cross-validation
- **Oversampling**: SMOTE, handling class imbalance
- **KNN**: distance-based prediction, optimal k selection, normalisation
- **Decision trees**: entropy, Gini index, pruning, depth selection
- **Ensemble methods**: bagging, random forests, boosting (XGBoost)
- **Naïve Bayes**: probabilistic classification, Laplace smoothing
- **Bayesian optimisation**: GP surrogate, acquisition functions (UCB, EI, PI)
- **Logistic regression**: binary/multiclass classification
- **SVMs**: kernel functions (RBF), soft-margin, classification for BBO region identification

### Coming in later modules (15–24)
- **Neural networks**: gradient descent, backpropagation, TensorFlow (Module 15)
- **Deep learning**: PyTorch, advanced architectures (Module 16)
- **CNNs**: convolutional neural networks (Module 17)
- **Hyperparameter tuning**: systematic methods and strategies (Module 18)
- **LLMs & transformers**: attention mechanisms, tokenisation (Modules 19–20)
- **Transparency & interpretability**: bias detection, model cards (Module 21)
- **Clustering**: hierarchical, k-means (Module 22)
- **PCA**: dimensionality reduction (Module 23)
- **Reinforcement learning**: multi-armed bandits, Q-learning, MDPs (Module 24)

### General strategies (always available)
- **Random search**: np.random.uniform — simple baseline
- **Grid search**: evaluate on a dense grid — good for low dimensions
- **Manual reasoning**: scatter plots, heatmaps, domain intuition
- **Perturbation**: search near the current best with small changes
- **Ensemble approaches**: combine multiple methods, take the best suggestion
- **Space-filling (balanced Voronoi)**: `max(min(dist_to_data, dist_to_boundary))` — penalizes corners. NEVER use raw Voronoi for BBO; it picks corners.
- **Latin Hypercube Sampling**: alternative space-filling
- **Y-weighted centroid**: average of top performers, weighted by output value
- **Model consensus**: trust models only on dimensions where multiple configs agree
- **Classifier + log-SVR decomposition (F1 primary)**: split into sign classification (SVM C=10) + log-space regression on log|Y|. Combined score = P(positive) × normalized log|Y|. Trust only if classifier LOO ≥85% AND candidate is far from known negatives. See `suggestions/f1_long_term_strategy.md`.
- **Linear model filtering**: exclude Ridge/SVR from ensembles when they extrapolate to boundary corners
- **NN surrogates** (`/train-nns`): pre-train MLPs per function, save to `models/week_XX/`. `/analyze` auto-loads them. Test 4 regularization variants (plain/dropout/weight-decay/ensemble) and 2 widths via 5-fold CV.

## Technical Notes

- **Colorblind-safe palettes**: always use coolwarm/viridis cmaps and Wong palette (#0072B2, #D55E00, #009E73, #E69F00, #CC79A7, #56B4E9)
- For 2D functions (F1, F2): scatter plots and heatmaps are powerful — visualise before deciding
- For higher dimensions: parallel coordinates, per-dimension correlations, top-vs-bottom boxplots
- Each function has different characteristics — one strategy won't fit all
- Track what works per function and adapt individually
- The summary cell should reference `next_query_N` variables from analysis cells — do NOT hardcode duplicate values
- Use LOOCV with RMSE (not R²) for model evaluation — R² is undefined for 1-sample LOO folds
- Suppress LOO warnings: `warnings.filterwarnings('ignore', message='R.*score is not well-defined')`
