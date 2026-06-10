# Week 11 Queries — Submitted 2026-06-10

## Formatted Queries

```
Function 1: 0.537663-0.537663
Function 2: 0.700902-0.948722
Function 3: 0.535311-0.606855-0.488771
Function 4: 0.365315-0.406648-0.425220-0.415740
Function 5: 0.997502-0.999923-0.906811-0.889405
Function 6: 0.437279-0.380214-0.541531-0.736888-0.021051
Function 7: 0.097614-0.290804-0.391792-0.214006-0.313924-0.776936
Function 8: 0.121695-0.188091-0.129244-0.172617-0.723946-0.741142-0.172353-0.619335
```

## Methods

| F | Method | W10 Result | Predicted / Note |
|---|--------|-----------|-------------------|
| F1 | **Gaussian-magnitude structural model — center bet** | 3.65e-7 (W10 exact-repeat confirmed W3 deterministic) | ln\|Y\| fits a concave quadratic R²=0.97 (Gaussian ridge along x1≈x2, center (0.537, 0.537), never sampled). Every LOO refit predicts \|Y(center)\| ≥ 6e3 — 10⁶× the largest observed \|Y\|. The 6-week-refined W3 peak is a molehill ~33 ln-units below the summit. Sign is the only open question: SVC P(+)=0.63 at center, a ~60/40 bet with extreme payoff asymmetry. Even a negative result calibrates the model amplitude and gives the most informative sign point on the board for W12. |
| F2 | **Noise-aware GP, argmax P(single draw > banked 0.6961)** | 0.436 (secondary region — not a real second peak) | **F2 confirmed STOCHASTIC, the only one of the 8.** Two exact/near repeat pairs (W6/W9 same X → 0.6961 vs 0.6251; W10 near-repeat → 0.436 vs 0.539) give σ ≈ 0.06, matched by the GP's learned WhiteKernel. With best-single-draw grading, the right acquisition is P(beat), not argmax-mean; all three acquisitions converge on the cluster interior (P(beat) ≈ 0.18, vs 0.02 for the secondary, 0.015 for unexplored). The cluster is the answer; each week is an ~18% lottery ticket. |
| F3 | **GP Expected Improvement — x3-ridge bet** (deviation from cluster-B) | -0.0264 (NEW BEST, cluster-B refinement) | Strategy investigation: cluster-B refinement saturated (W8→W10 +0.001, GPs predict -0.026±0.003 = zero expected gain). Model suggestions bimodal on x3 — trees at cluster B (0.07), the two best-CV GPs at x3≈0.49 with predicted POSITIVE Y (never seen in 26 pts). EI at the unexplored ridge is 44× a cluster-B step. F1-style ln(-Y) quadratic tested and rejected (R²=0.53, saddle). Downside bounded: trees say ≈-0.05 (mid-table). |
| F4 | **4-GP local consensus, x1 pinned** | 0.5524 (W10 exact-repeat confirmed W6 deterministic) | Investigation rejected: local quadratic (saturated, 9p/9pts), single-model GP trust (W8 -69% lesson — GP-M15-ARD leads CV +88.5% but is the family that crashed), second-peak search (global EI 5× below local). Cliff anatomy is empirical: x1 ±0.016 → -0.30/-0.38; x3 -0.023 → -0.011; \|Δ\|=0.008 → -0.0018. Query: 4 GP variants' local argmaxes inside \|Δ\|≤0.015, x1 pinned ±0.003. 3 of 4 predict ~0.61. Final \|Δ\|=0.0086, dominant move x3 (soft dim). |
| F5 | **★ TuRBO multi-kernel TS continuation** (winning kernel Matern15) | 7663.60 (NEW BEST, +25%) | 10 consecutive new bests (984 → 7663, 7.8×). Strategy validated every week — nothing changed (per instruction). State after applying W10 success: succ=1, fail=0, best=7663.60. W10 proved the summit is NOT the all-1s corner (x3 pulled to 0.955 and Y still jumped) — this draw continues the interior probe, pulling x3/x4 further back while holding x1/x2 at the corner. |
| F6 | **TuRBO RETIRED → 4-GP local consensus** (x5-down probe) | -0.4475 (5th consecutive regression) | **Deep-dive verdict: TuRBO structurally wrong for F6.** corr(distance-from-W6-best, Y) = -0.98 over all 30 points — a near-perfect unimodal pit. The 5 regressions (incl. 2 TuRBO bets at \|Δ\|=0.15/0.22) were guaranteed. Radial paraboloid confirms the shape (LOO +72%) but under-predicts the summit by 0.25. Replaced with 4-GP local consensus \|Δ\|≤0.035: all 4 GPs predict a new best, dominant move x5 0.049→0.021 (strongest-corr dim r=-0.74, never tested locally). Risk bounded by W7 precedent (-0.06 at same scale). |
| F7 | **TuRBO RETIRED → 4-GP local consensus** (W9-direction interpolation) | 1.6238 (regression — bold x3 jump failed) | TuRBO retired for step-size mismatch (not pure structure): F7 rewards directed 0.05-0.1 moves but TuRBO draws 0.2-0.5 (W9 -0.16, W10 -0.71). 4-GP local consensus \|Δ\|≤0.12: step is x1 +0.045, x3 +0.054 — interpolating toward W9's validated direction at half scale. 3 of 4 GPs predict new best (2.36-2.44 vs 2.33). W8-W9 midpoint rejected (no gain), global exploration rejected (EI 2.6× lower). |
| F8 | **Hybrid upgraded: 4-model consensus + ceiling warp, x6 anchored** | 9.9091 (plateau, within 0.002 of W6 best) | Per-dim hybrid kept (only strategy that holds the plateau 9.899-9.911), upgraded with: (1) ceiling transform ln(10.5-Y) fits R²=0.98 / LOO +78% — F8 approaches a ceiling; (2) 4-model local consensus (1 raw + 3 ceiling-warped GPs, all +73-89% LOO) \|Δ\|≤0.10, x6 pinned to 0.741 (every top-4 point). All 4 predict new best (9.93-9.94). Dominant move x5 up — continues the direction that made the W6 best. |

## Strategic theme this week

**Investigation drove real change — the standing instruction was "check if the current strategy is best," and the answer differed by function:**

- **Kept unchanged (validated)**: F5 TuRBO (10 straight bests), F2 cluster (now reframed as a stochastic P(beat) lottery).
- **TuRBO retired (2 functions)**: F6 (structural — unimodal pit, corr(dist,Y)=-0.98, large steps guaranteed to lose) and F7 (step-size mismatch — rewards 0.1-scale moves, TuRBO draws 0.2-0.5). Both replaced by local GP consensus.
- **New structural models**: F1 Gaussian-magnitude ridge (ln\|Y\| quadratic R²=0.97, last session) and F8 ceiling transform (ln(10.5-Y) R²=0.98). Both reveal exploitable shape the raw-Y models miss.
- **Deviation from saturated refinement**: F3 abandoned cluster-B (zero expected gain) for an EI bet on the unexplored x3-ridge (44× the EI).

**Cross-function convergence**: F4, F6, F7 independently arrived at the same multi-GP local-consensus pattern, each with a trust radius calibrated to that function's measured tolerance (0.015 / 0.035 / 0.12). This is the W11 generalization of the W8 "no single-model dominance" lesson — consensus across kernel variants, restricted to the locally-safe region.

## TuRBO state files updated

- `data/function_5/turbo_state.json`: succ=1, fail=0, best=7663.60 (W10 success applied; continuing).
- `data/function_6/turbo_state.json`: W10 result recorded then RETIRED — state archival only.
- `data/function_7/turbo_state.json`: W10 result recorded then RETIRED — state archival only.

## Notes

- Date: 2026-06-10
- Total cells in `notebooks/week_11.ipynb`: 71
- All queries computed from data + models, no hardcoded values; full notebook re-executes clean
- Plots saved to `plots/week_11/` (per-dim, parallel coords, F1 2D scatter + magnitude model)
- Running best Y per function: F1=3.65e-7 (W3), F2=0.6961 (W6), F3=-0.0264 (W10★), F4=0.5524 (W6), F5=7663.60 (W10★), F6=-0.1173 (W6), F7=2.3305 (W8), F8=9.9112 (W6)
- W10 produced 2 new bests (F3, F5), confirmed F1/F4 deterministic via exact-repeat, confirmed F2 stochastic
- W11 is the penultimate week — every decision now weighs "one more shot" payoff vs the safe close-out (documented in `suggestions/suggestions_for_week_12.md`)

## Sources Referenced This Week

- F1 structural model: `f(x) = h(x)·exp(quadratic)` decomposition (notebook Cell F2, `suggestions/f1_long_term_strategy.md`).
- TuRBO multi-kernel TS: Shibata et al. (Optuna BBO Challenge 2020). `src/turbo.py` — retired for F6/F7 this week.
- Ceiling / output warping for bounded targets: extends the Yeo-Johnson idea (`src/output_warping.py`) to an explicit ln(C-Y) for F8.
- Multi-GP local consensus: the W8 single-model-dominance lesson generalized to a trust-region consensus across kernel variants.
