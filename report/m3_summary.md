# M3 summary: does the penalty P behave the same for simulated annealing as for Gurobi's QUBO solver?

**Short answer: no.** On the paper's own ORLIB instances, single-flip simulated annealing (SA) is
sensitive to P where Gurobi's QUBO solver was not. Its best P is about 5% of the optimal cost, well
below the paper's 25–100% range. Inside that range, SA's solution quality gets steadily worse as P grows.

Data: `results/reproduction_20261110_215451.csv` (2,960 runs). Checks: `results/m3_verification_20261111_104144.txt`
(all pass). Tables: `results/m3_tables_20261111_104248.md`. Figures: `results/figures/m3_*.png`.

## Setup

| | This study | Li, Hennig, Kochenberger & Glover (2026) [P] |
|---|---|---|
| Model | QUBO of eq. 15, penalty-function construction (identical matrix to the Q-matrix construction; checked) | eq. 15, both constructions |
| Solver | classical simulated annealing (`dwave-samplers` 1.8.0, default schedule), 50 reads × 1,000 sweeps | Gurobi 10 QUBO solver (also D-Wave hybrid, NGQ) |
| Instances | A-1, A-2, B-1: 37 ORLIB files = **12 distinct problems** (files differ only in capacities, which the UFLP ignores) | A, B, C classes |
| Penalty | P = r × optimum, r ∈ {1, 2, 5, 10, **25, 50, 75, 100**}% (paper's values in bold) | r ∈ {25, 50, 75, 100}% |
| Seeds | 10 per (file, r) | – |
| Quality | gap = (cost − opt)/opt × 100: **raw** (best feasible read, as in [P]) and **repaired** (keep the opened facilities, assign each customer to its cheapest open one) | raw |
| Time | CPU seconds per run | Gurobi time to best solution |

Averages are taken over distinct problems (each first averaged over its files and seeds).

## Results

**1. Feasibility agrees with the paper, and it can now be explained.** At every r in the paper's range,
the lowest-energy read of all 1,480 runs is feasible ([P] §4.5.1: "no infeasibility is encountered even
for small penalty coefficient 25% for all the instances"). Below 25%,
no read is feasible at all. A short argument (`qubo.penalty_bounds`, tested on the paper's toy example)
shows why:
- the QUBO minimum is guaranteed feasible once P > max( max_j min_i (f_i + c_ij), max_i f_i );
- that bound is 16.6–22.6% of the optimum on these instances;
- so every QUBO the paper solved was provably exact;
- the jump in feasibility between 10% and 25% is a property of the model, not of the solver.

**2. Quality differs completely from Gurobi's 0% gap.** At the paper's 50%:

| | A-1 | A-2 | B-1 |
|---|---|---|---|
| raw gap (as in [P]) | 56.6% | 80.1% | 109.6% |
| repaired gap | 5.6% | 10.9% | 28.3% |
| [P] Gurobi QUBO | 0% | 0% | 0% |

No run, at any P, reached the optimum.

**3. SA's best penalty lies below the exactness threshold.** The mean repaired gap is lowest at r = 5% for
10 of the 12 problems (one at 10%, one at 25%, where the curve is flat). By class:

| r | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| A-1 | **4.0** | 4.4 | 5.1 | 5.6 | 5.9 |
| A-2 | **7.7** | 8.5 | 9.6 | 10.9 | 11.2 |
| B-1 | **13.9** | 17.0 | 23.4 | 28.3 | 32.1 |

At 5%, SA's raw reads are all infeasible, but the facilities they open are good. Over the paper's
range, a larger P is worse for SA. Gurobi's solution quality, by contrast, did not depend on P on these
classes (0% gap at all four values, [P] Table 2), though its time did: B-1 took 18.09 s at 25% against
about 1.7 s at 50–100% ([P] Table 3).

**4. More effort does not fix it.** On cap74, 10× more sweeps improved the repaired gap by at most
1.6 points, and not at all at r ≥ 25% (`results/sweeps_check_20261110_214251.csv`). This is consistent
with energy barriers: moving one customer between facilities passes through a state that costs about P,
and a single-flip search rarely crosses it when P is large.

**5. What makes an instance hard differs by solver.** All of these instances have a zero LP gap, which
is why Gurobi solved them quickly. For SA they are not easy, and difficulty rises with the fixed-cost
level: within each class, the problem with the fewest open facilities (cap74, cap104, cap134; 4 open)
is the worst. On B-1 at 50% the gap ranges from 13.6% (cap131) to 45.7% (cap134).

**6. Time.** CPU time per run is flat in r (fixed sweeps): about 2.1 / 3.8 / 12.5 s for A-1 / A-2 / B-1.
It is not comparable with [P] Table 3, which reports Gurobi's time to its best solution.

## What this means for the project

- **Research question 1 is answered.** A single rule for P does not transfer between solvers. For
  Gurobi, 25–100% all worked. For single-flip SA, about 5% is best and the paper's 50% is 1.6–14.4 points
  worse after repair.
- **For the ML extension:** the useful scale for P comes from the cost data (the exactness bounds), not
  from the optimum. The sufficient bound needs no optimum, so it is a natural feature and a natural
  baseline rule.

## Limitations (state these when reporting)

- Simulated annealing only, with one fixed setting (50 reads × 1,000 sweeps, default β schedule). These
  are classical results; they say nothing direct about quantum annealing or Gurobi.
- Classes A-1, A-2 and B-1 only: 12 distinct problems, all with a zero LP gap. B-2 (large gap) was not run.
- The repair step is ours; [P] needed none. Raw gaps are the like-for-like comparison.
- The machine slept for 11 h 46 min during the sweep. The six affected runs have inflated wall time
  only; results are seeded and the CPU times are unaffected.
