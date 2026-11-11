# M5 summary: can a good penalty be predicted from cheap features?

**Short answer: partly.**
- **Within the training distribution, yes.** On 200 synthetic instances, a learnt rule beats the best
  fixed penalty (mean repaired gap 1.69% vs 2.94%; paired 95% CI of the difference [0.86, 1.67]) and
  comes close to the per-instance oracle (1.30%).
- **On a held-out family with a large LP gap,** it avoids the collapse of the best fixed rule (6.7% vs
  45.2%). The simple bound-based rule does about as well there.
- **On ORLIB it does not transfer:** it is worse than a fixed 4%.
- **The model beats the paper's 50% and the tutorial's 100% by 30–52 points** of gap on the synthetic
  data.

Data: `results/ml_instances_20261111_112048.csv`, `results/ml_runs_20261111_112106.csv` (7,200 SA runs,
verified), and for ORLIB `results/reproduction_20261110_215451.csv` plus
`results/orlib_extra_20261111_*.csv` (600 extra runs). Full tables: `results/ml_eval_20261111_162203.md`.
Code: `src/uflp/ml.py`, `scripts/train_penalty_model.py`.

## Setup

- **Label.** r\* = P\*/LP bound: the grid value (12 values, 1–100%) with the lowest mean repaired cost
  over 3 seeds. A run with no solution counts as a failure, not as missing.
- **Model.** XGBoost regressor on log r\* (300 trees, depth 3, fixed hyperparameters, no tuning), on 10
  scale-free features, none of which uses the optimum:
  - size;
  - fixed-to-service ratio;
  - greedy–LP gap;
  - share of sites the greedy opens;
  - mean fixed cost / LP;
  - largest cheapest-service cost / LP;
  - the two exactness bounds / LP;
  - coefficients of variation of f and c.

  Predictions are snapped to the nearest grid value, and each rule's outcome is looked up from the
  existing runs, so no new SA runs were needed on the synthetic data.
- **Rules compared.** Everything tuned uses training instances only:
  - the model;
  - the best fixed r on the training set;
  - k × (sufficient exactness bound), with k tuned;
  - LP/n (the pilot's rule);
  - 50% of LP (the paper's choice, with an estimate);
  - 100% of LP (the middle of the tutorial's range);
  - the oracle r\*.
- **Statistics.** Paired bootstrap over instances (10,000 resamples) for the difference to the model.

## Results

**A. Stratified 5-fold CV, all 200 instances** (mean repaired gap %):

| model | best fixed r (4%) | k × bound | LP/n | 50% of LP | 100% of LP | oracle |
|---|---|---|---|---|---|---|
| **1.69** | 2.94 | 5.30 | 115.9 | 42.3 | 48.7 | 1.30 |

The model is better than the best fixed rule by 1.25 points [0.86, 1.67], and 0.39 points [0.27,
0.53] behind the oracle. LP/n (2%) is far too small here: it averages 116% because its repaired gaps
on the large-gap family exceed 400%.

**B. Leave one family out** (train on three families, test on the fourth):

| held-out family | model | best fixed r | k × bound | 50% of LP | oracle |
|---|---|---|---|---|---|
| **gap** (LP gap 1–26%) | 6.70 | 45.18 | 8.77 | 45.96 | 2.29 |
| uniform | 0.90 | 0.89 | 1.96 | 31.31 | 0.63 |
| euclid | 1.79 | 1.94 | 6.18 | 46.17 | 1.12 |
| orlib_like | 1.82 | 5.15 | 8.35 | 45.87 | 1.17 |

- On the unseen large-gap family the model is 38 points better than the best fixed rule [34, 43],
  but not distinguishable from k × bound (+2.1 [−1.9, +6.7]).
- On the other families it matches or beats the fixed rule.

**C. ORLIB sanity check** (trained on all 200; 12 problems, 10 seeds; here LP bound = optimum):

| model | best fixed r (4%) | k × bound | 50% (paper) | 100% | oracle |
|---|---|---|---|---|---|
| 9.91 | **8.15** | 9.94 | 14.95 | 16.41 | 7.73 |

The model predicts 3–4%, but ORLIB's best r is 4–15%, so it is **1.76 points worse than the fixed 4%**
[0.80, 2.82]. The synthetic families evidently miss something that matters on ORLIB. The
`orlib_like` family copies ORLIB's equal fixed costs and demand-scaled costs, so the difference lies
elsewhere; finding it is open.

## How to read this

- **The model is mostly a family detector.** The most important features are the coefficient of
  variation of c (35% of gain) and the largest cheapest-service cost relative to LP (21%), which
  separate the large-gap family. Within a family, its prediction tracks the oracle only weakly
  (Spearman 0.14–0.34), except on the large-gap family (0.53). It makes 2–3 distinct predictions per
  family.
- **The robust lesson is about scale, not the model.** For single-flip SA the useful P is a few
  percent of the LP bound, about one customer's service cost. The paper's and the tutorial's rules,
  which are right for Gurobi, cost 30–50 points of gap here. A per-instance rule matters mainly when
  instances differ in structure (here, the LP gap).
- **The oracle is optimistic.** It picks the best of 12 values using the same noisy seed averages it
  is scored on, so the model's distance to it is an upper bound on the real headroom.

## Limitations

- **SA only, at one setting** (50 reads × 1,000 sweeps). Labels depend on the solver and its effort.
- **200 instances in 4 families, n = 50 only.** The labels lie in a narrow band (2–6%), so there is
  little within-family signal to learn.
- **ORLIB is a sanity check only:** zero LP gap, 12 problems.
- **No hyperparameter tuning and no feature selection were done,** to avoid overfitting a small data
  set. A classifier over grid points was not tried.

## Follow-up: why the model does not transfer to ORLIB

Full tables: `results/ml_eval_20261111_171210.md` (five families, 250 instances).

**1. ORLIB lay outside the training data.** On the model's two most important features, ORLIB is
far beyond every synthetic family:

| feature | ORLIB | largest synthetic value |
|---|---|---|
| largest cheapest-service cost / LP | 0.14–0.22 | 0.083 |
| CV of service costs | 2.14–2.22 | 0.86 |

The cause is customer demand. ORLIB's demands run from 31 to 12,912 (CV 1.74), and one customer holds
22% of all demand. The synthetic families drew demand from 5–100 (CV about 0.5). Unit costs are
ordinary (CV about 0.55).

**2. Adding the missing property helps, but does not fix it.** I added a fifth family,
`heavy_demand`: lognormal demands with σ ∈ {1.0, 1.2, 1.4}, 50 instances, no ORLIB data used. It
reaches ORLIB's range (CV of costs up to 2.2). I then ran 1,800 more labelled SA runs, all checks
passing. Results:

| mean repaired gap % | model | best fixed r | oracle |
|---|---|---|---|
| 5-fold CV, 250 instances | **2.29** | 3.41 | 1.76 |
| `heavy_demand` held out | **4.66** | 5.26 | 3.60 |
| ORLIB (before → after) | 9.91 → 9.14 | **8.15** | 7.73 |

On ORLIB the model is still worse than a fixed 4%, by 0.99 points [0.12, 2.02].

**3. The remaining difference is the shape of the penalty curve.**

| mean gap % at r = | 3% | 4% | 5% | 8% | 15% | 25% |
|---|---|---|---|---|---|---|
| ORLIB (12 problems) | 10.35 | **8.15** | 8.51 | 9.64 | 10.85 | 12.72 |
| `heavy_demand` (50) | **4.51** | 5.26 | 7.22 | 14.29 | 26.70 | 36.43 |

- On the synthetic instances, SA's quality collapses quickly once P is too large. On ORLIB the curve
  is almost flat from 4% to 15%, and some problems prefer much larger P: cap101 does best at 15%
  (2.38% vs 4.25%, against a seed standard error of about 0.35).
- A model trained on steep curves learns to aim low, which on ORLIB lands on the steep side (3%).
- The features describe costs, not how SA's quality responds to P, so they cannot see this
  difference.

**What this means.**
- On ORLIB, a fixed 4% of the LP bound is the better rule.
- Within families, the model's predictions barely track the per-instance best (Spearman 0.04–0.23),
  except on the large-gap family (0.56). It mostly recognises families.
- I stopped tuning the generator here on purpose: tuning it further toward ORLIB would turn ORLIB
  from a check into training data.
- **Caveat.** The `heavy_demand` family was added *after* seeing ORLIB's failure, so ORLIB is no
  longer an untouched test set.
