# Suggestions for Week 09

Conditional strategies based on W8 query outcomes. Updated as `/analyze N` runs in W8.

---

## F1 — gradient-climb from W3 best (W8 deliberate deviation)

**W8 query**: (0.724297, 0.702040), step 0.025 from W3 best (0.7002, 0.6954) along the +/− gradient direction (away from the closest negative at (0.6501, 0.6815) Y=−3.6e-3).

**Method**: Branch 5 (strong-but-disagreeing) — single-model trust on KNN for the maxima-search question, since KNN was the only model whose argmax predicted Y > 0. The 3 GPs + GB were all valley-trackers (predicted "max" was Y ≈ −6e-4). 8/10 models beat baseline for the first time, but only KNN finds the positive zone.

**SVC classifier**: 82.35% LOO (still 2.6pp below 85% trust gate).

**Conditional strategies for W9:**

| W8 result | W9 strategy |
|-----------|-------------|
| Y > +3.65e-7 (improvement) | Real positive gradient confirmed. Continue stepping 0.030–0.040 in the same direction. Candidate ≈ (0.748, 0.708). Re-run KNN to see if its argmax has shifted. |
| Y ∈ [0, +3.65e-7] (zero plateau) | Positive island has plateau structure. Try perpendicular step from W3 best: candidate ≈ (0.690, 0.720) or (0.710, 0.680). |
| Y ∈ [−1e-30, 0] (vanishing negative) | Sign boundary is right at W8 candidate. Halve the step toward W3 best. Candidate ≈ (0.712, 0.699). Helps classifier learn boundary. |
| Y < −1e-3 (real negative) | Positive island is tiny — W3 best is a single isolated point, not a peak. Pivot to maxima-seeker ensemble (currently just KNN argmax = W3 best itself), or revisit if classifier crosses 85% with the new boundary point. |
| **Classifier crosses 85%** | Combined classifier+log-SVR path opens. Re-evaluate F1 long-term strategy. The combined-score argmax (currently 0.606, 0.596 — middle-zero zone) becomes a viable candidate. |

**Long-term reference**: `suggestions/f1_long_term_strategy.md`. The classifier trust gate is still the primary unlock for principled F1 queries.

---

## F2 — STRONG-consensus ensemble step

**W8 query**: (0.708074, 0.946424). Step +0.00444 in x1, −0.00051 in x2 from W6 best (0.7036, 0.9469).

**Method**: Branch 4 — RMSE-weighted ensemble of 6 interior models (RF, GB, GP-Matern05/15/25, GP-RBF). KNN dropped by outlier filter. All 10 models beat baseline; spreads 0.037 (x1) and 0.035 (x2) — both STRONG.

**Honest concern**: W7 used same approach (step 0.003) and regressed 17%. The ensemble step is small enough that if F2 has measurement noise of order ~0.1, this won't escape it.

**Conditional strategies for W9:**

| W8 result | W9 strategy |
|-----------|-------------|
| Y > 0.6961 (new best) | Step direction validated. Try +0.008 step in same x1 direction. |
| Y ∈ [0.6, 0.6961] | Marginal. Continue ensemble approach, accept the peak is at this region. |
| Y ∈ [0.45, 0.6] | **Likely measurement noise**. W9: **repeat W6 best (0.7036, 0.9469) exactly** to directly test the noise hypothesis. If Y comes back near 0.70, noise confirmed. If Y comes back ~0.5, the peak is at a different point we haven't tested. |
| Y < 0.45 | F2 has more structure than we modelled. Pivot to TuRBO or explore secondary cluster at (0.6658, 0.1240) Y=0.539. |

**Open question for W9**: should we sacrifice one query to repeat the W6 best and quantify F2's noise floor? Could be worth more than another tiny step.

---

## F3 — Warped-GB cluster B refinement (REVISED after outlier catch)

**W8 query**: (0.658278, 0.616243, 0.072861) — Warped-GB argmax on outlier-cleaned data.

**Method**: Branch 2 (one model dominates). After removing the outlier at (0.15, 0.44, **0.99**) Y=−0.40, r(x3,Y) collapsed from −0.544 to −0.097 (artifact). The earlier "smooth-GP extrapolation to x3 ≈ 0.50" was driven entirely by that one point. Without it, the 2 strongest models (GB +69%, Warped-GB +67%) both point to cluster B (x3 ≈ 0.07). GB rejected for suspect x2=0.914 (extrapolating, no other model agrees). Warped-GB chosen as strongest non-suspect candidate; its x2=0.616 is consistent with the 2nd-best data point at (0.6001, 0.7251, 0.0661) Y=−0.0364.

**Lesson logged**: F3 outlier sensitivity is now documented in Cell A — every future /analyze 3 run should print the WITH/WITHOUT correlation table to catch this kind of issue.

**Conditional strategies for W9:**

| W8 result | W9 strategy |
|-----------|-------------|
| Y > -0.035 (new best, cluster B richer than cluster A) | Pivot fully to cluster B. Try (0.62, 0.70, 0.06) close to the 2nd-best init point. |
| Y ∈ [-0.05, -0.035] (cluster B competitive) | Both clusters are similar plateaus. Try perpendicular refinement or revisit cluster A. |
| Y ∈ [-0.10, -0.05] (modest regression) | Cluster B peak sharper than thought. Refine W2 at (0.481, 0.533, 0.052) which gave Y=-0.040. |
| Y < -0.10 (3rd consecutive regression) | TuRBO trigger met (2 regressions + this one). W9 should use TuRBO on F3. |

---

## F4 — GP-Matern15 dominant-model step

**W8 query**: (0.350866, 0.417803, 0.439220, 0.444161) — GP-Matern15 argmax.

**Method**: Branch 2 (one model dominates). GP-Matern15 at RMSE 0.81 (+91.2%) beats runner-up GP-Matern05 by a 49.9% relative RMSE gap. All 4 dims have STRONG ensemble consensus (spreads 0.04–0.06), but the best-model argmax differs from the ensemble by ~0.02 on x1 and x4, in a direction that matches the NN gradient at W6 best: [-12.8, +10.1, -1.9, +5.0]. Step |Δ|=0.036 (~3× the ensemble step), which is bolder but better-informed than W7's |Δ|=0.008 flat result.

**Conditional strategies for W9:**

| W8 result | W9 strategy |
|-----------|-------------|
| Y > 0.5524 (new best) | GP-Matern15 + NN gradient direction confirmed. Continue same direction, step 0.03-0.04. |
| Y ∈ [0.50, 0.5524] (plateau) | F4 is at its peak. Switch to even smaller step or repeat-best noise test. |
| Y ∈ [0.0, 0.50] (regression) | GP-Matern15 direction was wrong. Fall back to ensemble step. |
| Y < 0 (sharp peak miss) | F4 peak is sharper than estimated; second regression after W7 flat. TuRBO trigger. |

---

## F5 — TuRBO multi-kernel TS continuation

**W8 query**: (0.385930, 0.965134, 0.999466, 0.965273) — TuRBO-1 multi-kernel TS, winning kernel **Matern15**.

**Method**: Branch 6 (TuRBO continuation). W7's deliberate deviation succeeded with +26% (2669 → 3365). State machine: L=0.8, succ_counter=1, fail=0, best=3365.22. This is the first use of the multi-kernel TS upgrade (Optuna BBO 2020 paper) — TuRBO drew samples from all 4 kernels (Matern 0.5/1.5/2.5, RBF) and picked the argmax across the (kernel, candidate) grid.

**Trajectory**: 984 → 1207 → 1412 → 1979 → 2308 → 2669 → 3365 (every week a new best). x3 has been monotonically increasing toward 1.0; TuRBO pushed it to 0.9995.

**Standard pipeline alternative (informational)**: boundary-consensus hybrid would give (0.43, 0.97, 0.99, 0.95) with |step|=0.048 — different bet (push x1 instead of x3). TuRBO chosen because the state machine continues naturally from W7's success.

**Conditional strategies for W9:**

| W8 result | W9 strategy + state |
|-----------|---------------------|
| Y > 3365 (continued climb) | succ_counter → 2. L still 0.8. Continue TuRBO. |
| Y ∈ [3000, 3365] (marginal) | fail_counter +=1. Continue TuRBO; TR stays at L=0.8. |
| Y ∈ [2000, 3000] (modest regression) | fail_counter continues incrementing. If consecutive fails reach 4, L halves. |
| Y < 2000 (significant drop) | TR shrinks faster. Consider switching back to standard ensemble for one week to break out of TR. |

**Note**: state file `data/function_5/turbo_state.json` has been pre-updated with the W7 result. W9 `/analyze 5` will load this state, apply W8's actual Y on top, and proceed.

---

## F6 — STRONG ensemble consolidation (reverse direction from W7)

**W8 query**: (0.386326, 0.363766, 0.545708, 0.735494, 0.047722) — RMSE-weighted ensemble of 7 interior models.

**Method**: Branch 4 — all 5 dims STRONG consensus (spreads 0.015-0.136). GB dropped by outlier filter (x1 disagreed). SVR top at +76%, GP-Matern05 +74%. NN gradient at best [-0.57, -0.57, -1.28, -1.39, -2.04] confirms all dims should DECREASE — ensemble step (-0.034, -0.013, +0.008, -0.004, -0.001) does this on 4 of 5 dims. W7 stepped the OPPOSITE direction on x1 and regressed 52%; W8 reverses.

**Conditional strategies for W9:**

| W8 result | W9 strategy |
|-----------|-------------|
| Y > -0.1173 (new best) | Direction validated. Continue ensemble approach with step ~0.04. |
| Y ∈ [-0.18, -0.1173] (matches W7 or improves) | Regression streak broken. Continue ensemble approach. |
| Y ∈ [-0.30, -0.18] (similar plateau) | F6 stuck around -0.2. Try perpendicular step direction. |
| Y < -0.30 (2nd consecutive regression from peak) | TuRBO trigger met. W9 switch to TuRBO on F6. |

---

## F7 — TuRBO triggered by 2-regression rule

**W8 query**: (0.052374, 0.528292, 0.073612, 0.216500, 0.316123, 0.779613) — TuRBO multi-kernel TS, winning kernel **Matern05**.

**Method**: Branch 6 (TuRBO). Framework rule "2 consecutive regressions" triggered: W5 (1.6078) → W6 (1.4147, −12%) → W7 (1.1157, −21%, total −31% from peak). State machine: L=0.8, succ=0, fail=1 after applying W7 result.

**Why TuRBO over standard ensemble**: standard ensemble step is |0.16| with x5=0.194 — below the entire top-5 x5 range [0.317, 0.420]. Models are extrapolating beyond observed-good territory. TuRBO instead anchors tightly on x1, x4, x5, x6 (TR tight from short lengthscales) and explores on x2, x3 (TR full [0,1] from long lengthscales). Winning kernel Matern05 cross-validates with sklearn CV (GP-Matern05 also top sklearn model at +68%).

**Conditional strategies for W9:**

| W8 result | W9 strategy + state |
|-----------|---------------------|
| Y > 1.6078 (new best) | TuRBO exploration paid off. succ_counter → 1. Continue. |
| Y ∈ [1.4, 1.6078] (matches W5/W6) | TR exploration didn't hurt much. fail_counter → 2. Continue TuRBO. |
| Y ∈ [1.1, 1.4] (similar to W7 regression) | TuRBO didn't help. fail_counter → 2. Consider one query at top-4 centroid to re-anchor. |
| Y < 1.1 (3rd straight regression) | F7 deeply stuck. Force TuRBO restart (state.restart_count++) or pivot to top-4 centroid pullback. |

---

## F8 — TuRBO multi-kernel TS (deliberate plateau-break)

**W8 query**: (0.008646, 0.016274, 0.188833, 0.352216, 0.793490, 0.863726, 0.146039, 0.933019) — TuRBO-1 multi-kernel TS, winning kernel **Matern05**.

**Method**: Branch 6 (TuRBO). **Deliberate framework deviation** — F8 hasn't strictly met the 2-regression rule (W7 was -0.012 from W6, only one sub-best result). After 3 weeks of plateau at ~9.9 with the hybrid suggesting tiny refinement steps, switched to TuRBO to explore deliberately. The bet: small steps in a plateau aren't going anywhere; better to take a real exploratory action.

**Trust-region shape**:
- x1-x4, x7: TR moderately tight (lengthscales shorter), candidate near W6 best
- x5, x6, x8: TR very wide (long lengthscales), TS draws large exploration steps
- Multi-kernel TS picked candidates from Matern05's posterior — rougher than sklearn CV's preferred Matern25 (+91%). This is the multi-kernel TS exploring an alternative prior.

**State after applying W7 result**: L=0.8, succ=0, fail=1, best=9.9112.

**Risk acknowledged**: candidate pushes x5 to 0.79 (top-7 range [0.49, 0.71]), x6 to 0.86 (top-7 ~0.74), x8 to 0.93 (top-7 [0.59, 0.75]). Multiple dims pushed beyond observed-good territory. Nvidia's TuRBO paper specifically warned about this failure mode when GP lengthscales are too long.

**Conditional strategies for W9:**

| W8 result | W9 strategy + state |
|-----------|---------------------|
| Y > 9.9112 (new best, plateau broken) | TuRBO bet paid off. succ=1. Continue TuRBO. |
| Y ∈ [9.85, 9.9112] (similar to plateau) | TR exploration didn't help, didn't hurt. fail=2. Continue TuRBO; TR will shrink. |
| Y ∈ [9.5, 9.85] (fell off plateau) | TuRBO over-explored. fail=2. Switch back to hybrid for W9 to re-anchor. |
| Y < 9.5 (large drop) | Force TR restart. Fall back to top-4 centroid for W9. |
