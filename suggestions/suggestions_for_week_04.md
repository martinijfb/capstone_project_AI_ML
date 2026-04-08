# Suggestions for Week 04

Collected from week 03 decision markdown cells and analysis.

## Per-Function Recommendations

### F1 (2D, 13 pts)
- Week 03 query: classifier + log-SVR combined candidate at ~(0.70, 0.70)
- If Y is **positive and > 7.7e-16**: the classifier approach worked — refine near this point
- If Y is **~0 or negative**: begin **Phase 1 (space-filling)** from `suggestions/f1_long_term_strategy.md` — Voronoi largest empty circle for 3-4 weeks to build coverage, then return to classifier approach with better data
- First Voronoi target: ~(0.66, 0.44), radius 0.245
