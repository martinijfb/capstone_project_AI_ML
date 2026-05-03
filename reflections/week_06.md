# Week 06 Reflection: Module 17 (CNNs)

## Background notes (not for submission)

- Word limit: 700 (portal). Five prompts about progressive feature extraction, breakthroughs vs incremental, depth/cost/overfitting trade-offs, CNN building blocks, and benchmarking success.
- Anchored on: expanded GP family (Matern smoothness as a hyperparameter), F1 classifier crossing 85% threshold, F2 narrow peak step-sizing, F8 plateau-break attempt, F1 informed exploration framing.
- Q5 grounded in actual Dunbar transcript: hierarchical intelligence system (binary check → CNN wake-up at ~hundreds of microwatts in CSEM's always-on people-recognition device).

## Submission (plain text, ~695 words)

**Progressive feature extraction**

The "edges to textures to objects" analogy fits how my picture of each function has built up. Round 1 was edges: which inputs matter, where the bad regions live. By round 6 it feels more like textures: F1 is near-zero across most of its space, F4 has a peak so narrow that small steps overshoot, F5 wants three of its four inputs pushed high.

A concrete example. Each week I fit several model families (random forest, gradient boosting, SVR, Gaussian processes, a small neural net) and ask each one where the optimum is. They sometimes converge tightly on a dimension and sometimes scatter. F7's "weak dimension" (the one where they scatter most) has rotated. It was x2 in week 4, x6 in week 5, and now only x1 and x3 are tight enough to trust the model average. The picture localises round by round, like deeper CNN layers latching onto more specific patterns.

**LeNet to AlexNet, breakthroughs vs incremental**

Most weeks have been small adjustments. The closest thing to a step-change this round was structural. My Gaussian Process surrogate uses a Matern kernel, with a parameter ν that controls how smooth the fit is: small ν allows sharp wiggles, large ν forces smoother surfaces. I had been using ν=2.5 and RBF for weeks. This round I added ν=0.5 (rough) and ν=1.5 (intermediate), so smoothness becomes something cross-validation picks instead of something I hard-code.

The effect surprised me. On F4, F5, F6, and F7 the rougher variants either lead my model rankings or tie at the top. Only F8 still prefers the smoother kernels, fitting given it's the highest-D function with a smooth landscape. The takeaway mirrors the LeNet to AlexNet jump: I didn't add a new technique, I just stopped fixing one of my hyperparameters.

**Depth, cost, overfitting (explore vs exploit)**

The trade-off was visible this week. F2's peak is sharp, so I picked a step of about 0.007 from the current best, the smallest step I've taken on any function. F8 has plateaued for three rounds, so I went the other way and picked the most aggressive query of the project, moving one dimension by 0.17 because both my model ensemble and the neural network's gradient at the current best pointed the same way. The CNN parallel writes itself: a deeper kernel captures sharper features but overfits, an aggressive query finds new ground but risks regressing. I size each step by how confident the model fits look on CV.

**Convolutions, pooling, activations, loss**

Activation functions changed my framing the most. On F1 I run a sign classifier (an SVM predicting whether a candidate will return positive or negative), but I only let it influence my query when its cross-validation accuracy crosses 85%. This week it crossed for the first time after weeks at around 70%, opening a gate that had been closed all project. The same logic gates individual dimensions everywhere else: when several models agree tightly on where to push a dimension, I take their average; when they scatter, I fall back to a weighted average of the best points I've already seen. Either it fires or it falls back, more directly useful than loss or pooling.

**Edge deployment, benchmarking success**

Dunbar's example, the always-on people-recognition device at CSEM that runs a cheap binary "is anyone there?" check before waking up the full CNN, maps directly onto how my pipeline already works. Every candidate goes through cheap filters first (boundary check, outlier check, classifier trust gate for F1), and the expensive ensemble averaging only runs on what passes. The cascade is the whole point in her low-power chip and the whole point in mine.

The constraint also reshapes what success means. With one query per function per week, I can't measure against a global maximum I don't know. I evaluate each query against three things: did it improve Y, did the model's prediction calibrate against the actual return, did it move me into a region the framework couldn't have ruled out otherwise. F1 hasn't improved in five rounds, but its sign classifier went from 64% to 87%. That's refining the cheap stage of a cascade even when the expensive one isn't ready to fire.

## Submission (HTML version for portal)

```html
<p><strong>Progressive feature extraction</strong></p>
<p>The "edges to textures to objects" analogy fits how my picture of each function has built up. Round 1 was edges: which inputs matter, where the bad regions live. By round 6 it feels more like textures: F1 is near-zero across most of its space, F4 has a peak so narrow that small steps overshoot, F5 wants three of its four inputs pushed high.</p>
<p>A concrete example. Each week I fit several model families (random forest, gradient boosting, SVR, Gaussian processes, a small neural net) and ask each one where the optimum is. They sometimes converge tightly on a dimension and sometimes scatter. F7's "weak dimension" (the one where they scatter most) has rotated. It was x2 in week 4, x6 in week 5, and now only x1 and x3 are tight enough to trust the model average. The picture localises round by round, like deeper CNN layers latching onto more specific patterns.</p>

<p><strong>LeNet to AlexNet, breakthroughs vs incremental</strong></p>
<p>Most weeks have been small adjustments. The closest thing to a step-change this round was structural. My Gaussian Process surrogate uses a Matern kernel, with a parameter ν that controls how smooth the fit is: small ν allows sharp wiggles, large ν forces smoother surfaces. I had been using ν=2.5 and RBF for weeks. This round I added ν=0.5 (rough) and ν=1.5 (intermediate), so smoothness becomes something cross-validation picks instead of something I hard-code.</p>
<p>The effect surprised me. On F4, F5, F6, and F7 the rougher variants either lead my model rankings or tie at the top. Only F8 still prefers the smoother kernels, fitting given it's the highest-D function with a smooth landscape. The takeaway mirrors the LeNet to AlexNet jump: I didn't add a new technique, I just stopped fixing one of my hyperparameters.</p>

<p><strong>Depth, cost, overfitting (explore vs exploit)</strong></p>
<p>The trade-off was visible this week. F2's peak is sharp, so I picked a step of about 0.007 from the current best, the smallest step I've taken on any function. F8 has plateaued for three rounds, so I went the other way and picked the most aggressive query of the project, moving one dimension by 0.17 because both my model ensemble and the neural network's gradient at the current best pointed the same way. The CNN parallel writes itself: a deeper kernel captures sharper features but overfits, an aggressive query finds new ground but risks regressing. I size each step by how confident the model fits look on CV.</p>

<p><strong>Convolutions, pooling, activations, loss</strong></p>
<p>Activation functions changed my framing the most. On F1 I run a sign classifier (an SVM predicting whether a candidate will return positive or negative), but I only let it influence my query when its cross-validation accuracy crosses 85%. This week it crossed for the first time after weeks at around 70%, opening a gate that had been closed all project. The same logic gates individual dimensions everywhere else: when several models agree tightly on where to push a dimension, I take their average; when they scatter, I fall back to a weighted average of the best points I've already seen. Either it fires or it falls back, more directly useful than loss or pooling.</p>

<p><strong>Edge deployment, benchmarking success</strong></p>
<p>Dunbar's example, the always-on people-recognition device at CSEM that runs a cheap binary "is anyone there?" check before waking up the full CNN, maps directly onto how my pipeline already works. Every candidate goes through cheap filters first (boundary check, outlier check, classifier trust gate for F1), and the expensive ensemble averaging only runs on what passes. The cascade is the whole point in her low-power chip and the whole point in mine.</p>
<p>The constraint also reshapes what success means. With one query per function per week, I can't measure against a global maximum I don't know. I evaluate each query against three things: did it improve Y, did the model's prediction calibrate against the actual return, did it move me into a region the framework couldn't have ruled out otherwise. F1 hasn't improved in five rounds, but its sign classifier went from 64% to 87%. That's refining the cheap stage of a cascade even when the expensive one isn't ready to fire.</p>
```
