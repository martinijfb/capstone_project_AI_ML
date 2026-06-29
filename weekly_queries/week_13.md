# Week 13 Queries (Final Round) — Submitted 2026-06-18

## Formatted Queries

```
Function 1: 0.551199-0.512321
Function 2: 0.695271-0.953066
Function 3: 0.507777-0.613369-0.470253
Function 4: 0.359205-0.419152-0.418101-0.415250
Function 5: 0.999999-0.999999-0.999999-0.999999
Function 6: 0.420967-0.374172-0.538399-0.736854-0.059911
Function 7: 0.153662-0.255553-0.410900-0.210633-0.271749-0.774594
Function 8: 0.099669-0.179799-0.140168-0.108990-0.837949-0.739510-0.168593-0.651190
```

## Methods

| F | Method | W12 Result | Predicted / Note |
|---|--------|-----------|-------------------|
| F1 | **Off-node perpendicular jackpot swing** | -0.00229 (W12 envelope shot went negative) | The W12 negative revealed the structure: the magnitude peak near the diagonal centre is a NEGATIVE basin; the centre is a tiny positive on a zero of the sign-carrier. A large positive can only come from inside that small positive island, off the zero-node. Query steps 0.03 off the envelope centre along the perpendicular (unsampled) axis: P(+)≈0.65, envelope \|Y\|~4e5, safe 0.09+ from negatives. Bank (3.65e-7) is locked, so this is a max-upside free swing — ~60% sign chance at something 100x-4000x the bank. |
| F2 | **GP posterior argmax over the cluster** | 0.5596 (below bank) | F2 is near-maxed: the bank (0.7205) sits above every model's predicted surface (peak mean ~0.59). The cluster is the only competitive region (secondary tested and lost W10). Final query = the highest-value novel point the GP identifies inside the cluster — the best remaining shot at a true peak at/above the bank. WhiteKernel used only to regularize the one repeated input. |
| F3 | **Ridge-local 4-model consensus** | -0.00189 (NEW BEST, +62%) | W12 improved the ridge again by stepping x1-down; the refit continues that gradient (x1 -0.012, x3 -0.013), \|Δ\|=0.018. The two raw GPs now extrapolate a zero-crossing just SW — a first-ever positive Y would be the new best. Y is approaching its ceiling at 0, so absolute upside is small but real. |
| F4 | **Refit 4-GP local consensus, x1 pinned** | 0.6883 (NEW BEST, +1.7%) | Third straight consensus win. This step is led by x2 (+0.013), the soft dim the models now favour, x1 held at the cliff value. All 4 GPs predict a new best (0.69-0.81). Risk bounded: top-3 within 0.016, all ≥ 0.55. |
| F5 | **Extreme corner** | 8633.9 (NEW BEST, +12.7%) | W12 settled it: the function climbs monotonically to (1,1,1,1) — every push toward the corner gained, no regression anywhere. All 4 GPs place their max at the exact corner (~8655 vs banked 8634). Final query is the format-cap supremum on every dim — the rare fully-justified boundary optimum. Bank locked at 8634, so zero downside. |
| F6 | **3-GP micro consensus, x5-up** | -0.1160 (NEW BEST — dethroned the 6-week W6 summit) | The W12 micro-step finally beat the W6 summit by stepping x5-up; the gradient is confirmed (x5 0.021→-0.231, 0.054→-0.116). Refit continues x5-up another +0.006, \|Δ\|=0.006. All 3 healthy GPs predict a new best (-0.107 to -0.112). RBF-ARD stays ejected (lengthscale collapse). The radial pit means anything larger than a micro-step loses. |
| F7 | **Refit 4-GP local consensus, x1-up** | 2.6183 (NEW BEST, +4.5%) | Third straight consensus win; x1-up has driven all three (x1 0.053→0.098→0.135, Y 2.33→2.51→2.62). Refit continues x1-up (+0.019), \|Δ\|=0.035. 3 of 4 GPs predict a new best (2.61-2.65); rough Matern05 cautious as before. Second-best point only 0.048 away, so a miss stays 2.5+. |
| F8 | **Refit ceiling-warp consensus, x6 anchored, x5-up** | 9.9400 (NEW BEST) | x5-up has paid four rounds running (x5 0.50→0.67→0.72→0.80, Y 9.868→9.940, a monotone ladder). Final query continues to x5≈0.838, x6 pinned at 0.741 (every top point), \|Δ\|=0.057. Ceiling structure ln(10.5−Y) R²≈0.98 still holds; warped family is 3 of 4 voters. All 4 predict bank-or-above (9.940-9.941). |

## Strategic theme — the final round

**This is the last query round of the project (Module 24).** Every query is pure exploitation: bank the best defensible point per function, with no contingency beyond it. W12 produced 6 new bests, and the final round follows the validated gradients:

- **5 functions (F3, F4, F6, F7, F8) continue their W11/W12 winning consensus directions, recentred** — the multi-GP local-consensus method (born on F4 in W11) is the workhorse, each with a trust radius and a measured gradient.
- **F5 takes the confirmed boundary optimum** — the extreme corner, the rare case where the data proves the maximum sits on the boundary.
- **F2 is near-maxed** — the bank exceeds all model predictions, so the query is the best novel point in the only competitive cluster.
- **F1 is the one genuine gamble** — the bank is locked and unbeatable by refinement, so the final query is a maximum-upside swing at the high-magnitude positive island, accepting the sign risk because there is nothing to lose.

**Two functions where the bank stands unbeaten this round regardless of outcome:** F1 (3.65e-7, a sign-carrier problem the data can't fully resolve in 22 points) and F2 (0.7205, a function whose peak the models place below the banked lucky value).

## Per-function final bests (entering this query)

F1=3.65e-7 (W3) · F2=0.7205 (W11) · F3=-0.00189 (W12) · F4=0.6883 (W12) · F5=8633.9 (W12) · F6=-0.1160 (W12) · F7=2.6183 (W12) · F8=9.9400 (W12). Six of eight peaked in W12, the project's strongest round.

## Notes

- Date: 2026-06-18 (final round, Module 24)
- Total cells in `notebooks/week_13.ipynb`: 69; full notebook re-executes clean (no errors)
- All queries computed from data + models, no hardcoded values
- Plots saved to `plots/week_13/`
- TuRBO fully retired across all functions by W12; the multi-GP local-consensus method and the two structural models (F1 Gaussian-magnitude envelope, F8 ceiling transform) carried the endgame.

## Sources Referenced This Week

- F1 structural envelope `f = h(x)·exp(quadratic)` with envelope/h-zero separation (`suggestions/f1_long_term_strategy.md`).
- Ceiling transform ln(C−Y) for bounded-from-above targets (F8; C=0 variant on F3).
- Multi-GP local consensus across kernel variants — the dominant endgame method.
