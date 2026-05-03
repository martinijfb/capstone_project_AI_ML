# Suggestions for Week 07

Conditional next-step strategies based on Week 06 query outcomes.

---

### F1 (2D, 16 pts after W06 query)
- **W06 query**: balanced Voronoi at **(0.617191, 0.222274)** — Q4, the most under-sampled positive-only quadrant (tied with Q2 at 2 pts each)
- Decision branch: #1 (no models beat baseline; combined classifier + log-SVR runs but trust check fails on `d_to_neg`)
- **What changed in W6**: Classifier LOO finally crossed 85% (87% with SVC C=1) — first time the trust gate is open. But the combined candidate (~0.64, 0.63) sits 0.05 from the negative at (0.65, 0.68), failing the ≥0.15 gate. Log-SVR calibration error at best is 2.4 (acceptable).
- **Why Q4 not Q3**: Q3 has 3 negatives — querying there generates more boundary data but doesn't help find peaks. Q4 has 2 positives only, genuinely unexplored, higher chance of new local maximum.
- **If W6 Y is positive (≥0)**: Q4 is benign → next week target **Q3 balanced Voronoi (0.274, 0.309)** for negative-region characterisation, OR re-run combined approach with d_to_neg gate raised to 0.20 (with the new Q4 positive added, the boundary should be cleaner)
- **If W6 Y is largely negative (< -1e-3)**: 5th negative observation; log-SVR will recalibrate substantially. Re-run combined approach next week with full data — d_to_neg gate may now produce a candidate that passes, since the negative region is better-characterised
- **If W6 Y is LARGE positive (> 0.01) — EXIT CONDITION**: switch to exploitation around (0.617, 0.222); we found a peak region
- Long-term: continue per `f1_long_term_strategy.md` until classifier LOO ≥ 85% (✓ achieved W6) AND combined candidate stays > 0.15 from any negative (✗ still failing) AND log-SVR calibration improves (✓ within 5)

**Peer-strategy test log:**
- W6: tested Matern ν=0.5 (Athanasios reported it tripled his F1 best). Result for our data: all 3 Matern variants {0.5, 1.5, 2.5} tie at LOO RMSE 0.001955 (none beat baseline). With 15 pts and 2 large negatives dominating, the GP collapses to mean prediction regardless of smoothness. **However**, Matern 0.5 on rank(Y) and combined-with-classifier independently picked ~(0.64, 0.27) — same Q4 region as Voronoi. Useful as a cross-check, not as a primary model.
- Worth re-testing Matern ν=0.5 once we have ≥20 pts and ideally a non-negligible positive — the kernel may unlock once there's a real spike to fit.

### F2 (2D, 16 pts after W06 query)
- **W06 query**: RMSE-weighted ensemble of KNN/RF/GB at **(0.703636, 0.946935)** — STRONG consensus on both dims (spread 0.011 / 0.033). Distance from current best 0.0066.
- Decision branch: #5 (strong consensus, both dims STRONG, hybrid degenerates to ensemble)
- W5 outcome: ensemble's 0.012 step from best returned Y=0.500 (drop from 0.666). Peak is sharp — smaller step this week.
- **Notable**: All 4 GPs (Matern 0.5/1.5/2.5, RBF) fail baseline this week (-7.1%, all tied). GP family doesn't fit F2's structure with 15 pts. Continue using KNN/RF/GB ensemble; don't include GP in F2's ensemble until it actually beats baseline.
- NN gradient at best: dx1=+0.69, dx2=+0.41 (says push higher on both — disagrees with ensemble's slight x2 decrease)
- **If W6 Y improves (>0.666)**: ensemble is calibrated at this finer scale — continue same approach next week
- **If W6 Y in 0.45-0.66**: still climbing the same peak; reduce step further by midpointing W6 query with current best
- **If W6 Y drops (<0.45)**: ensemble is poisoned by W5 query (0.694, 0.963, Y=0.500) — exclude it as outlier-candidate and re-fit, OR explore the (0.666, 0.124) low-x2 cluster which holds Y=0.539

### F3 (3D, 21 pts after W06 query)
- **W06 query**: RMSE-weighted ensemble of RF + GB at **(0.418642, 0.591747, 0.333934)** — STRONG consensus on all 3 dims (spread 0.027 / 0.013 / 0.021). Distance from current best 0.077.
- Decision branch: #2 (GB dominant at 60.8% improvement, ~2× over RF; both interior and tightly agreeing → ensemble effectively GB-weighted)
- W5 outcome: GB-suggested (0.515, 0.547, 0.340) returned Y=-0.0470 (slight regression from best -0.0348) — flat plateau confirmed
- **Continued outlier handling**: keep training on no-outlier subset (drop pt at (0.15, 0.44, 0.99) Y=-0.40). The full-data baseline (0.0763) inflates artificially; clean baseline is 0.0335.
- **GP family observations**: All 4 GPs (Matern 0.5/1.5/2.5, RBF) tied at -5.6%, none beat baseline. 3rd function in a row where GP family fails (F1, F2, F3 all). GP needs more data to fit small-magnitude landscapes.
- **NN observation**: NN trained on full data (with outlier inflating its baseline) — looks bad against cleaner no-outlier baseline. Re-training NN on no-outlier subset for W7 might help.
- NN gradient at best: dx1=-0.05, dx2=+0.09, dx3=+0.53 (says push x3 UP — disagrees with ensemble keeping x3 stable; NN has been excluded so this is informational only)
- **If W6 Y improves (> -0.0348)**: GB exploitation worked → continue tightening around the new best next week
- **If W6 Y in -0.05 to -0.035**: still on the plateau → try the RF alternative direction (low-x3 cluster around 0.61, 0.78, 0.08)
- **If W6 Y drops (< -0.05)**: GB direction is wrong → pull back to midpoint of best and W6 query, OR refit with W6 as outlier-candidate

### F4 (4D, 36 pts after W06 query)
- **W06 query**: RMSE-weighted ensemble of 7 interior models at **(0.366863, 0.404869, 0.433503, 0.415759)** — STRONG consensus on all 4 dims after NN outlier-filter (spread 0.098 / 0.051 / 0.066 / 0.058). Distance from current best 0.028.
- Decision branch: #5 (many models, all dims STRONG → pure ensemble)
- W5 outcome: ensemble's 0.028 step from best returned Y=0.4055 (regression from 0.5414). Same step size this week but different direction (x3 ↑ instead of ↓).
- **NEW: Matern 1.5 leads at +73.0%** — first function where the additional Matern variants (0.5, 1.5) materially change the leaderboard. Previously dominated by Matern 2.5/SVR ensemble. Worth tracking as a permanent improvement.
- **GP-RBF FAILS baseline** for the first time on F4 (-2.9%) — the smoother RBF kernel can't fit F4's sharp narrow peak structure.
- NN outlier-filtered (suggestion (0.028, 0.473, 0.537, 0.603) is far from the cluster). NN gradient at best: dx1=-11.8, dx2=+19.8, dx3=+0.7, dx4=-1.6 (push x1 down, x2 up — disagrees with ensemble).
- **If W6 Y improves (> 0.541)**: ensemble was right — peak found. Continue same step size (~0.028) next week.
- **If W6 Y in 0.3-0.5 (plateau again)**: 2nd consecutive 0.028-step plateau → cut step to ~0.014 (midpoint between W6 query and current best) for W7.
- **If W6 Y < 0 (overshoot)**: pull back to GP-Matern15's solo pick or revert to current best with tiny step.
- **If W6 Y << -1 (catastrophic)**: revert to pure exploitation of current best.

### F5 (4D, 26 pts after W06 query)
- **W06 query**: hybrid at **(0.354155, 0.921140, 0.968749, 0.947704)** — ensemble for x1/x2/x4 (STRONG consensus), REFINED boundary-consensus for x3 (max of top-5 max 0.952 and ensemble 0.969 = 0.969). Distance from current best 0.030.
- **Boundary-consensus rule was refined this week** (made permanent in framework): for high-edge consensus, clip to `max(top-5 max, interior ensemble)` instead of just top-5 max. For low-edge: `min(top-5 min, interior ensemble)`. Preserves safety against extrapolation while not freezing dims that valid models support stepping.
- Decision branch: #5 (STRONG consensus all dims; boundary-consensus on x3)
- W5 outcome: NEW BEST 2307.5 (+17% over W4 1979). F5 has improved every single week W1→W5.
- **Boundary-consensus on x3 fired**: 3 non-Ridge models (RF, GB, NN) all push x3 > 0.98, correlation r(x3,Y)=+0.666 matches → clip to top-5 max 0.952. **NN gradient on x3 (+5941) is the strongest of all 4 dims** — there's tension between the framework's safety cap and NN's signal that we should push higher.
- **NEW: GP-Matern05 +79.2%** is the top model (ahead of NN +77%). 2nd function where rough Matern wins. **Matern 2.5 and RBF FAIL** baseline (-4.2%) — smoother kernels can't fit F5.
- **If W6 Y > 2307 (improvement)**: continued climbing → consider RELAXING the boundary-consensus cap on x3 in W7. Test ensemble x3=0.969 (interior, supported by 4 valid models) instead of cap 0.952.
- **If W6 Y similar (1900-2307, plateau)**: x3 cap is too restrictive → in W7 OVERRIDE boundary-consensus, use ensemble x3=0.969. Document as a deliberate exception when the rule conservatively caps a dim that's still climbing.
- **If W6 Y drops (< 1900)**: peak is between W5 and W6 → bracket via per-dim midpoint of W5 (0.336, 0.907, 0.952, 0.938) and W6 query (0.354, 0.921, 0.952, 0.948).
- **Peer gap reminder**: Sterling reached 5079, Nick reached 2496. We're climbing but slower. Worth investigating GP-UCB or Expected Improvement acquisition functions if W6 plateaus.
- **F5 NN gradient at best**: dx1=+4078, dx2=+3335, dx3=+5941, dx4=+5863 — all strongly positive, x3 strongest.

### F6 (5D, 26 pts after W06 query)
- **W06 query**: RMSE-weighted ensemble of 7 interior models at **(0.420022, 0.376592, 0.537773, 0.739730, 0.048710)** — STRONG consensus on all 5 dims (spread 0.186 / 0.124 / 0.134 / 0.143 / 0.019). Distance from current best 0.066.
- Decision branch: #5 (STRONG consensus all dims; no boundary-consensus this week)
- W5 outcome: NEW BEST -0.260 (+14% over W4 -0.304). Trajectory: W2 -0.44 → W3 -0.31 → W4 -0.30 → W5 -0.26.
- **NEW**: SVR +60.1% leads, **GP-Matern05 +54.5%** is 2nd (3rd function where Matern 0.5 beats Matern 2.5). **GP-RBF FAILS** baseline for 3rd consecutive function (F4, F5, F6).
- **Boundary-consensus update from W5**: x5 was at 2/3 threshold last week; now NN is outlier-filtered and SVR's x5 = 0.053 (above threshold). No boundary fires this week.
- NN gradient at best: dx1=-0.38, dx2=-0.23, dx3=-1.32, dx4=-0.62, dx5=-1.20 (push everything down). Ensemble agrees on x2/x4/x5 but disagrees on x3 (says UP). NN is informational only (outlier-filtered).
- Step from best is 0.066 — larger than W5's 0.036. If this overshoots, W7 should pull back.
- **If W6 Y > -0.26 (improvement)**: ensemble continues to work — same approach next week
- **If W6 Y in -0.35 to -0.26 (plateau)**: peak is local; try GP-Matern05's solo pick as alternative direction next week
- **If W6 Y drops (< -0.40)**: ensemble overshot — pull back to midpoint between W6 query and current best

### F7 (6D, 36 pts after W06 query)
- **W06 query**: hybrid at **(0.031697, 0.474149, 0.142789, 0.217730, 0.335014, 0.787502)** — boundary-consensus REFINED on x1, ensemble on x3 (STRONG dims), top-4 centroid on x2/x4/x5/x6.
- Decision branch: #4 (weak global consensus, STRONG only on x1 and x3, plus boundary-consensus on x1)
- W5 outcome: NEW BEST 1.608 (+7.7% over W4 1.493). Trajectory: W2 1.365 → W3 1.461 → W4 1.493 → W5 1.608.
- **NEW**: GP-Matern05 +57.8% leads (4th function — F4/F5/F6/F7 all). **GP-Matern25 + GP-RBF FAIL together** (-2.9% each, 4th consecutive function). At this point GP-RBF should be considered for removal from future grids.
- **Boundary-consensus on x1 fired**: 3 non-Ridge models (RF, SVR, GP-Matern15) push x1 < 0.02, r=-0.56 matches. REFINED rule = min(top-5 min 0.0354, ensemble 0.0317) = 0.0317 (slightly below top-5 min — refinement allows interior agreement to push past the safety floor).
- **Big move on x3**: ensemble pushes x3 from 0.220 → 0.143 (-0.077). All 4 valid models cluster tightly (spread 0.047). NN gradient on x3 = -1.38 (agrees with ensemble direction).
- NN gradient at best: dx1=-1.88 (✓), dx2=+0.14 (centroid stable), dx3=-1.38 (✓), dx4=+0.87 (centroid stable, NN says push up), dx5=-1.24 (centroid stable, NN says push down), dx6=+0.02 (stable)
- Step from current best is 0.084 — larger than W5's 0.052.
- **If W6 Y > 1.608**: x3 push down was correct → continue same approach
- **If W6 Y in 1.4-1.6 (plateau)**: x3 over-extrapolated → use midpoint x3 ≈ 0.18 in W7
- **If W6 Y < 1.4**: ensemble overshot on x3 — pull back to current best's x3=0.220 with smaller perturbation
- **NN gradient hints for next-week pivot if needed**: if x3 push fails, try x4 UP (NN +0.87) or x5 DOWN (NN -1.24) instead

### F8 (8D, 46 pts after W06 query)
- **W06 query**: hybrid at **(0.155378, 0.200562, 0.075482, 0.215712, 0.672482, 0.740636, 0.179584, 0.616336)** — ensemble on x1/x2/x3/x4/x5/x8 (STRONG), top-4 centroid on x6/x7 (moderate). No boundary-consensus.
- Decision branch: #5 (STRONG on 6/8 dims, moderate on 2)
- W5 outcome: NEW BEST 9.868 (marginal +0.03% over 9.865). F8 has been on plateau — we made the most aggressive query of the project (step 0.27 from best) to break out.
- **Big move**: ensemble pushes x5 from 0.503 → 0.672 (+0.169), backed by NN gradient (dx5=+0.97, strongest of all 8 dims).
- **NEW pattern**: All 10 models beat baseline this week (first time). **F8 reverses F4-F7 pattern**: smoother kernels (Matern25 +75.1%, RBF +72.2%) win, while Matern05 is mid-pack. Suggests 8D + 45 pts has enough smoothness signal for the smoother kernels.
- Boundary-rejected: Ridge, SVR (top model excluded), GP-Matern05, NN. GB outlier-filtered.
- NN gradient at best: dx1=+0.48, dx2=-0.21, dx3=+0.66, dx4=-0.17, dx5=+0.97, dx6=+0.45, dx7=-0.24, dx8=-0.29
- **If W6 Y > 9.87 (improvement)**: x5 push paid off — peak is in the new region; continue same approach next week
- **If W6 Y in 9.5-9.87 (similar/slight regression)**: x5 step too aggressive but direction was right → in W7 use midpoint x5 ≈ 0.59
- **If W6 Y < 9.5 (catastrophic)**: ensemble misled by spurious correlations → revert to current best with small perturbation
- **Long-term**: F8 has 45 pts and clearly plateauing. Best testbed for trying GP-UCB / Expected Improvement acquisition functions next week — would address the "model says continue exploiting but Y won't move" pattern.

**Deferred peer-derived option (NOT pursuing this week):**
- Athanasios's W4 reflection mentions "F5's best point at (0.999, 0.004, 0.999, 0.999)" with "outputs jump from hundreds to thousands". This is a different basin from where we're climbing (high x1, low x2 vs. our low x1, high x2). Sterling reached Y=6204 in W5 (vs our 2307).
- **Why deferred**: (0.999, 0.004, 0.999, 0.999) is a corner pick — all 4 dims at boundaries. Our framework explicitly excludes such suggestions because models often hallucinate optima at boundaries via extrapolation. Without an interior model agreeing, this is just a peer's anecdote about a possibly-spurious model output.
- **When to revisit**: if our climbing trajectory plateaus (W6 or W7 fails to improve), spend ONE query at ~(0.95, 0.05, 0.95, 0.95) — interior version of Athanasios's claim, less boundary risk. If Y > 1000, the second basin is real and we pivot. If Y < 500, our climbing basin is the right one.
- Document this as a W7/W8 fallback, not a current week deviation.
