# Week 10 Queries — Submitted 2026-06-02

## Formatted Queries

```
Function 1: 0.700201-0.695377
Function 2: 0.665800-0.123969
Function 3: 0.642756-0.608923-0.070174
Function 4: 0.366878-0.404857-0.433485-0.415743
Function 5: 0.989254-0.999371-0.955453-0.999767
Function 6: 0.346282-0.207080-0.591232-0.827685-0.011716
Function 7: 0.016801-0.304938-0.816642-0.130663-0.330186-0.759414
Function 8: 0.167842-0.214880-0.083215-0.221848-0.686932-0.740753-0.224993-0.641579
```

## Methods

| F | Method | W9 Result | Predicted / Note |
|---|--------|-----------|-------------------|
| F1 | **Deliberate noise-test: repeat W3 best exactly** | +1.13e-10 (positive, not best — W3's 3.65e-7 still holds) | 6 weeks of failed refinement around W3. SVC classifier trajectory 71% → 82% → 83% → **84.21% (W10)**, only 0.8pp below the 85% trust gate. Transform research this week: `log10|Y|` (magnitude only) fits at +68.8% improvement vs raw Y's +47.4% — magnitude is smooth, sign is chaotic. `signed_log10|Y|` extrapolates to +1e90 (unusable as argmax). Rank transform +43%. The diagnosis is consistent: F1 is a sign-classification problem masquerading as regression. Repeat W3 to confirm reproducibility before committing to the combined classifier+log-SVR path next week. |
| F2 | **Deliberate noise-test: secondary peak (0.6658, 0.124)** | 0.6251 — confirmed σ≈0.05 noise on repeat of W6 best (W6 gave 0.6961, W9 gave 0.6251 at same X) | Main cluster noise now quantified numerically. Refinement within |Δ|<0.1 of the main cluster is indistinguishable from measurement noise. The secondary region (init Y=0.539) is the only untested alternative in 9 weeks; its Y is within noise distance of the main cluster. Models predict ~0.34 there but they only have 1 data point in that low-x2 region — extrapolation, not signal. |
| F3 | RMSE-weighted ensemble of interior models (3 dims STRONG) | -0.0274 (NEW BEST W8, +21%) | W8 cluster B refinement vindicated. W9 step |Δ|=0.07 went too far (Y dropped to -0.099). W10 step |Δ|≈0.02, half of W9's. 11 models beat baseline; Warped-GB at +66% leads, GP-Matern25 predicts new best at ~Y=-0.025. Outlier check confirmed init point at x3=0.090 inflated x3 correlation (r=-0.56 → -0.10 without it). |
| F4 | **Deliberate noise-test: repeat W6 best exactly** | 0.1710 → -54% from W6 best, third consecutive regression | Peak radius confirmed < 0.010 across 3 weeks: W7 stayed at |Δ|=0.008; W8 cliff at |Δ|=0.036 (-69%); W9 at |Δ|=0.020 (-54%). We have NEVER repeated W6 exactly. Spending W10 on the foundational reproducibility question — same X, same Y? — is more informative than another step that the data shows the geometry can't support. F2/F1 pattern. |
| F5 | **★ TuRBO multi-kernel TS continuation, winning kernel RBF** | 6125.59 (NEW BEST W9, +71% jump) | 9 consecutive new bests on TuRBO. State after applying W9 success: succ=1 (one more bumps L from 0.4 to 0.8). Multi-kernel TS picked **RBF kernel** again (W9 also RBF, W8 Matern15, W7 single-kernel) — kernel diversity matters. Step pushes 3 of 4 dims toward boundary. Trajectory: 984 → 6125 in 9 weeks (6.2× growth). |
| F6 | **★ TuRBO multi-kernel TS continuation, winning kernel Matern15** | -0.3812 (4th consecutive regression from W6's -0.117) | State machine self-corrected: fail counter advanced. failure_tolerance=6, room remains before L halves to 0.4. Multi-kernel TS picked Matern15 this week. Bold step (|Δ|≈0.46) — TR still wide-open on x4. F6 is the failure mode TuRBO was designed for: trust-region narrows as more regressions accumulate. |
| F7 | **★ TuRBO multi-kernel TS continuation, winning kernel Matern15** | 2.1727 (small regression from W8 breakthrough of 2.33) | State succ=0, fail=2 after applying W9 result. failure_tolerance=6, plenty of room. Multi-kernel TS picked Matern15. The state machine used W9's information: step explores x3 high region (+0.44 from W8 best — bold) while keeping other dims small adjustments. Different exploration direction from W9's failed x2-up attempt. |
| F8 | Per-dim hybrid (Branch 5) — 5 STRONG dims = ensemble, 3 moderate dims = top-4 centroid | 9.9106 (within 0.0005 of W6 best, 9.9112) | **W9 hybrid pull-back from W8 plateau-break worked exactly as planned**. 10/10 models beat baseline; GP-Matern25 leads at 92.2% improvement, GP-RBF at 91%. Explicitly NOT trusting single-model dominance even at 92% after the W8 F4 lesson (49.9% margin → -69% crash). Key data anchor: x6=0.741 in ALL top-4 points — centroid wins for x6 regardless of ensemble spread. Step |Δ|≈0.06 from W6 best, ≈0.09 from W9 best. |

## Strategic theme this week

**3 TuRBO continuations (F5/F6/F7), 2 noise-tests (F1/F4), 3 standard pipelines (F2/F3/F8)** — the noise-test pattern is now generalising:

- **F1 noise-test**: 6 weeks of small steps near W3 best (3.65e-7) all crashed by 30× to 3200×. Cannot reach 85% classifier gate yet. Spending one query to verify W3 is reproducible before betting on the classifier path next week.
- **F2 noise-test**: W9 already proved σ≈0.05 numerically. W10 tests the only remaining alternative region — the secondary peak — that no model has data to predict.
- **F4 noise-test**: W6 peak radius confirmed < 0.010 across 3 weeks of refinement failures. The peak might be a fluke; repeating exact coordinates is the only way to find out.

**3 TuRBO continuations**:
- F5: 9 consecutive new bests, kernel choice varies week to week (RBF/Matern15/RBF again). The multi-kernel ensemble keeps proving its value.
- F6: TuRBO state machine handling 4-regression streak as designed.
- F7: TuRBO state machine using last week's information to explore a different direction.

**Two standard wins**:
- F3: ensemble step pulls back from W9 overshoot with outlier-cleaned models.
- F8: hybrid recovered to within 0.0005 of W6 best last week; small refinement step this week.

**Lessons being applied this week**:
- Single-model dominance margins (even +92% on F8 GP-Matern25) are not trusted — the W8 F4 failure baked this into Cell E logic.
- Noise quantification (F2 W9) generalises: F1 and F4 get noise-tests this week.
- TuRBO is now the standard tool, not the exception, when refinement loops fail.

## TuRBO state files updated

- `data/function_5/turbo_state.json`: succ=1 after applying W9 success.
- `data/function_6/turbo_state.json`: fail counter advanced after W9 regression.
- `data/function_7/turbo_state.json`: succ=0, fail=2, best=2.33.

## Notes

- Date: 2026-06-02
- Total cells in `notebooks/week_10.ipynb`: 72
- All queries computed from data + models, no hardcoded values
- Plots saved to `plots/week_10/` (per-dim scatter, parallel coords, F1 2D scatter + combined-score plot)
- Running best Y per function: F1=3.65e-7 (W3), F2=0.6961 (W6), F3=-0.0274 (W8★), F4=0.5524 (W6), F5=6125.59 (W9★), F6=-0.1173 (W6), F7=2.3305 (W8), F8=9.9112 (W6, W9 within 0.0005)
- W9 produced 2 new bests (F5 +71%, classifier improvement on F1), 1 near-tie (F8 within 0.0005), 4 regressions (F2/F4/F6/F7), 1 outlier discovery (F3 init point inflating x3 correlation)
- Multi-kernel TuRBO state: same kernel diversity continues (RBF for F5 again, Matern15 for F6 and F7).

## Sources Referenced This Week

- F1 transform research: log10|Y|, signed_log10|Y|, rank, tanh; conclusion that F1 is a sign-classification problem.
- TuRBO multi-kernel TS: Shibata et al. (Optuna BBO Challenge 2020). `src/turbo.py`.
- Multi-model consensus over single-model dominance: F4 W8 lesson now applied to F8 W10 (92% dominance not trusted alone).
