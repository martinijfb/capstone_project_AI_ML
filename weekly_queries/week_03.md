# Week 03 Queries — Submitted 2026-04-09

## Formatted Queries

```
Function 1: 0.700201-0.695377
Function 2: 0.701898-0.953338
Function 3: 0.518808-0.622752-0.177963
Function 4: 0.404794-0.431684-0.402527-0.451256
Function 5: 0.270475-0.875752-0.915748-0.875603
Function 6: 0.387036-0.373428-0.513896-0.853071-0.046985
Function 7: 0.035368-0.480332-0.260021-0.206835-0.373160-0.775247
Function 8: 0.154668-0.195916-0.085056-0.247030-0.487307-0.747539-0.346852-0.750537
```

## Methods

| F | Method | Week 02 Result |
|---|--------|----------------|
| F1 | Classifier + log-SVR combined | ~0 (no improvement) |
| F2 | RMSE-weighted avg of KNN/RF/GB | 0.4265 (did not improve) |
| F3 | Y-weighted centroid of top 4 | -0.0400 (did not improve) |
| F4 | SVR suggestion (~70% LOOCV improvement) | -1.3905 (GP overshot) |
| F5 | RMSE-weighted model avg (all beat baseline) | 1206.76 (NEW BEST) |
| F6 | SVR suggestion (~55% LOOCV improvement) | -0.6086 (did not improve) |
| F7 | Hybrid: centroid + model override on STRONG dims (x1, x5) | 0.9318 (did not improve) |
| F8 | Hybrid: centroid + model override on STRONG dims (x1, x3, x4) | 9.8651 (NEW BEST) |
