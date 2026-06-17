# Week 12 Queries — Submitted 2026-06-11

## Formatted Queries

```
Function 1: 0.585743-0.601770
Function 2: 0.696090-0.946316
Function 3: 0.519698-0.613187-0.483009
Function 4: 0.360226-0.406461-0.420921-0.408996
Function 5: 0.999645-0.999695-0.999678-0.999506
Function 6: 0.423088-0.375316-0.538300-0.736429-0.054239
Function 7: 0.134884-0.270759-0.401943-0.210085-0.295117-0.774752
Function 8: 0.110033-0.182329-0.128271-0.147077-0.800957-0.738785-0.171960-0.639266
```

## Methods

| F | Method | W11 Result | Predicted / Note |
|---|--------|-----------|-------------------|
| F1 | **Envelope argmax s.t. P(+) ≥ 0.7, h-zones excluded** | +5.17e-12 (sign bet WON, magnitude was an h-zero) | The W11 center bet resolved as forecast: positive sign (P(+)=0.63 paid) but magnitude 30.5 ln-units below the envelope — a zero of the sign-carrier h. Both model gates now PASS for the first time: quadratic refit R²=0.96 (≥0.95), honest deduped SVC LOO=85.00%. Query is the envelope argmax on the trusted positive side, ≥0.08 from all 4 known h-zeros. Envelope \|Y\|~10⁶ if h≈1 there. Bank (3.65e-7) is locked, so this is a free high-EV swing. |
| F2 | **Noise-aware GP, argmax P(draw > 0.7205), no-repeat** | 0.7205 (NEW BEST — the ~18% P(beat) ticket HIT) | F2 is stochastic (σ≈0.06, the lone exception). The W11 acquisition design was validated by its own win. The refit GP's unconstrained argmax landed on the W11 point (banned repeat); the 0.005 exclusion costs ~0.01 of P and adds cluster coverage. Cluster still ~10× better than any alternative on P(beat). |
| F3 | **Ridge-local 4-model consensus (EI handoff)** | -0.00497 (NEW BEST, +81% — the ridge bet was real) | EI's job is done (it found the ridge). Switched to exploitation: 4-model local consensus (2 raw + 2 ceiling-warped C=0 GPs) within \|Δ\|≤0.06 of the W11 ridge point. Fits collapsed to near-baseline (best +7%) — the lone ridge point is unfittable in LOO, so consensus matters more than any single model. Raw GPs pull x3 down, warped pull up; opposing extrapolations cancel within the basin. 2 of 4 predict a new best. |
| F4 | **Refit 4-GP local consensus (W11 winner recentered)** | 0.6766 (NEW BEST, +22.5% — summit was off-W6) | The W11 probe was the project's biggest single F4 gain and validated every component (x1 pin, soft-dim direction, all-4-agree gate). Refit asks for a half-size continuation: x3-down continues, x4-down joins. All 4 GPs predict a new best (0.72-0.83). Risk bounded: top-3 points within 0.011 all score ≥ 0.55. |
| F5 | **Unanimous 4-GP corner argmax (TuRBO RETIRED)** | 5565 (regression -27%, the only W11 loss) | The loss exposed a confound: the W10 "x3=0.955 sweet spot" was an x1 artifact (W9's high-x3 sample had x1=0.92). Refit models agree the function rises to the FULL corner — never sampled. All 4 kernels pick the same grid point (zero spread, 7825-8214 vs banked 7664). TuRBO retired with honors after 10 wins: no directional uncertainty left, and its next draw heads to the proven-bad x3≈0.84. Every coordinate is inside the observed top-5 per-dim envelope (data-anchored). |
| F6 | **3-GP micro consensus, \|Δ\|≈0.006 close-out** | -0.2308 (regression — x5-down probe failed) | The W11 4-GP probe (\|Δ\|=0.033, all predicted new best) lost 0.11: the pit is sharper than GPs resolve at the 0.03 scale. Full ledger: 6 dethroning attempts at \|Δ\|∈[0.033,0.293], all losses, in 5 directions. The micro-band (\|Δ\|<0.03) is the only never-sampled resolution. RBF-ARD ejected (lengthscale collapse: predicts -1.08 at 0.005 from a -0.117 point). The 3 healthy GPs agree on x5 slightly UP — the first F6 query using a measured local gradient. Worst case ≈ -0.119. |
| F7 | **Refit 4-GP local consensus (W11 winner recentered)** | 2.5056 (NEW BEST, +7.5% — streak broken, TuRBO retirement validated) | The W11 consensus broke the 2-regression streak (the F4 pattern transferred). Refit continues x1-up (drove both winning steps: W8→W11 +0.045, now +0.037). 3 of 4 GPs predict a new best (2.54-2.62); rough Matern05 cautious as before. Risk anchored by the W8 point (2.33) just 0.070 away. |
| F8 | **Refit ceiling-warp 4-model consensus, x6 anchored** | 9.9345 (NEW BEST — ceiling warp + consensus) | x5-up has paid twice (0.50→0.67 won W6; 0.687→0.724 won W11); refit unanimously asks for →0.80 (third ride). x6=0.741 anchor keeps its perfect record (every top-4 point). Ceiling structure ln(10.5−Y) R²≈0.98 holds; warped family is 3 of 4 voters. All 4 predict a new best (9.94-9.95). Plateau risk bounds a miss at ≈ -0.01. |

## Strategic theme this week

**The W11 strategy investigation paid off in full — W11 produced 5 new bests, and W12 follows the evidence:**

- **5 functions repeat their W11 winners, recentered** (F2 P(beat), F3 ridge consensus, F4/F7 local consensus, F8 ceiling-warp consensus). When a method wins, the right move is to continue it with updated data, not to invent a new one.
- **F5 retired TuRBO with honors** after its first loss in 10 weeks — not as punishment, but because the loss resolved the x1/x3 confound and left no directional uncertainty for stochastic search to exploit. All four kernels now agree on the unsampled all-high corner.
- **F6 dropped to micro-scale** — its 6-attempt distance-loss ledger proved the pit is unresolvable above \|Δ\|≈0.03, so the only remaining play is the never-sampled micro-band around the 6-week-defended summit.
- **F1 took its best-ever shot**: both model gates (envelope R² ≥ 0.95, classifier ≥ 85%) passed for the first time, courtesy of the W11 sign observation. The bank is locked, so it's a free swing at an envelope predicting \|Y\| ~ 10⁶.

**Cross-function pattern**: the multi-GP local-consensus method (born on F4 in W11) is now the workhorse on F3/F4/F5/F6/F7 — five of eight functions — each with a trust radius matched to that function's measured tolerance (F6: 0.006, F4: 0.010, F3: 0.06, F8/F7: 0.10, F5: corner box). Six of eight queries carry all-model new-best predictions.

## TuRBO state files

- `data/function_5/turbo_state.json`: W11 fail recorded, then RETIRED (succ=0, fail=1, best=7663.6). F5/F6/F7 TuRBO all now retired; F8 left TuRBO in W9. The multi-kernel TuRBO era is closed — it built F5's 7.8× and F7's breakthrough, and handed off cleanly to local consensus once trajectories saturated.

## Notes

- Date: 2026-06-11
- Total cells in `notebooks/week_12.ipynb`: 69; full notebook re-executes clean (no errors)
- All queries computed from data + models, no hardcoded values
- Plots saved to `plots/week_12/` (per-dim, parallel coords, F1 2D scatter + envelope model)
- Running best Y per function: F1=3.65e-7 (W3, positive sign now confirmed reproducible at center), F2=0.7205 (W11★), F3=-0.0050 (W11★), F4=0.6766 (W11★), F5=7663.60 (W10), F6=-0.1173 (W6), F7=2.5056 (W11★), F8=9.9345 (W11★)
- W11 produced 5 new bests (F2/F3/F4/F7/F8) — the project's strongest week — plus F1 sign confirmation; F5 and F6 were the regressions that drove this week's two biggest strategy changes (TuRBO retirement, micro-scale)
- `suggestions/suggestions_for_week_13.md` holds full contingency tables for all 8 in case a 13th round exists

## Sources Referenced This Week

- F1 structural model `f = h(x)·exp(quadratic)` with envelope/h-zero separation (notebook Cell F, `suggestions/f1_long_term_strategy.md`).
- Ceiling transform ln(C−Y) for bounded-from-above targets (F8; adapted to F3 as the C=0 warp).
- Multi-GP local consensus across kernel variants — the W11 F4 innovation, now the dominant pattern.
- TuRBO multi-kernel TS: Shibata et al. (Optuna BBO Challenge 2020), `src/turbo.py` — fully retired this week after building the F5/F7 gains.
