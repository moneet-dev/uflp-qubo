# Capacitated extension: capacity constraints and slack variables in the QUBO

**Short answer.** The capacitated model of Li (2026, Urban Lifeline, Appendix A) can be put into
QUBO form exactly with binary slack variables. How the capacity row is written then matters more
than the penalty value:

- **Written literally** (Σ_j x_ij + s_i = k·y_i), single-flip SA cannot close facilities and stays
  16–26% from the optimum at every P.
- **Written without y** (Σ_j x_ij + s_i = k; linking still enforced by eq. 15's x ≤ y penalty), the
  QUBO is just as exact. SA then behaves as in the uncapacitated case: best near P = 5–10% of the
  optimum, with gaps of 7.7–9.4%, against 8.3% uncapacitated on the same problems and seeds.

These are classical simulated-annealing results only.

Code: `src/uflp/capacitated.py`, tests `tests/test_capacitated.py` (24 tests). Data:
`results/capacitated_20261111_170845.csv` (coupled, 960 runs) and
`results/capacitated_decoupled_20261111_175554.csv` (decoupled, 576 runs). Checks and tables:
`results/capacitated_analysis_*.md`, `results/capacitated_forms_20261111_190947.md`.

## Model and checks

- **CFLP with single sourcing,** written with a general weight: Σ_j w_j x_ij ≤ Q_i y_i.
  - w = 1, Q = k is [UL] (A.4): at most k customers per facility.
  - w = demand is the demand-capacity version of [UL]'s worked example and of the ORLIB cap files.
- **The exact MILP reproduces all seven optimal values of [UL] Table 3** (433.85, 377.28, 377.28,
  349.02 × 4), and which facilities are open. This needs single sourcing (split demand gives
  405.56 at capacity 40) and unrounded Euclidean distances (the table shows √2 as 1.41, √20 as 4.47,
  …).
- **QUBO.** The inequality becomes an equality with an integer slack s_i ∈ [0, Q_i], written in a
  bounded binary expansion (weights 1, 2, 4, …, plus a last weight that makes the maximum exactly
  Q_i). The equality is then penalized as P·(row / mean weight)².
  - An exhaustive search on a small instance shows the QUBO minimum equals the CFLP optimum, with
    the capacity binding, for both forms of the row.
  - On [UL]'s example, the energy of the MILP optimum equals its cost.
- **Why ORLIB's own capacities are not used.** Under single sourcing they are infeasible for most
  files: one customer's demand, 12,912, exceeds capacities of 5,000 and 10,000. ORLIB's published
  capacitated optima allow split demand, which a binary QUBO cannot express. The experiment
  therefore uses (A.4) on the 12 distinct ORLIB cost matrices with k ∈ {5, 10}: 24 problems. The
  capacity binds on all 12 problems at k = 5 and on 8 of 12 at k = 10; it costs up to 7.2% over the
  uncapacitated optimum. The QUBOs have 864–2,750 variables (48–200 of them slack).
- **Data checks pass for both sweeps:**
  - every (problem, k, r, seed) present;
  - no feasible or repaired cost below the CFLP optimum;
  - CFLP optimum ≥ UFLP optimum;
  - 6 sampled runs re-executed identically.

## Results

**Mean repaired gap %** over 12 problems (seeds 0–2, same SA settings as M3: 50 reads × 1,000 sweeps):

| r = P/opt | 1% | 2% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| uncapacitated (same problems and seeds) | 25.4 | 14.6 | **8.3** | 9.6 | 12.8 | 14.9 | 16.3 |
| k = 5, coupled | **16.6** | 19.9 | 21.8 | 21.9 | 22.5 | 22.6 | 22.5 |
| k = 5, decoupled | 68.3* | 10.0 | 9.4 | **9.4** | 11.6 | 12.9 | 14.4 |
| k = 10, coupled | **24.4** | 25.3 | 25.9 | 25.6 | 26.1 | 26.1 | 26.2 |
| k = 10, decoupled | 12.6 | 9.1 | **7.7** | 8.2 | 9.2 | 11.2 | 12.4 |

\*At r = 1% the overload penalty is smaller than the cost of opening another facility, so SA opens
too few. Repair cannot fit 50 customers at 5 per facility: 1,557 of 1,800 reads fail, and such runs
score 100%.

- **The decoupled form is better at every r ≥ 2%,** by 8–18 points. The paired 95% bootstrap
  intervals over the 12 problems all exclude zero.

**1. Coupled: facilities get locked open.** SA's reads open nearly every facility (15–16 of 16 on
cap74, 38–50 of 50 on cap134) where the optimum opens 6–10. The mechanism:

- When a facility's load plus slack equals k, flipping y_i from 1 to 0 changes the capacity row by
  k, which costs k²·P (25P for k = 5) in a single step.
- So facilities are never closed, repair has to pay every fixed cost, and the gap hardly depends on
  P. Note that this is the literal form of (A.4), the natural way to write it.

**2. Decoupled: the penalty behaves as without capacity.** Removing y from the capacity row loses
nothing, because a closed facility still cannot serve anyone, thanks to the x ≤ y penalty. It
removes the k² barrier, and the curve takes the uncapacitated shape again:

- the best P is about 5–10% of the optimum;
- too small a P fails, now through capacity violations;
- too large a P slowly hurts.

**3. Slack encoding has its own barriers.** At large P, with k = 5 (slack weights 1, 2, 2), 87–100%
of reads are feasible. With k = 10 (weights 1, 2, 4, 3), only 1–11% are, in either form.

- Changing the slack by one unit can require flipping two bits, for example removing the 4 and
  adding the 3. In between, the row is off by 3–4 units, which costs 9–16·P.
- SA settles for cheaper violations instead; on cap71 the reads double-assign about 11 customers.
- Repair still recovers good solutions, but the raw QUBO output is rarely usable. This is a concrete
  instance of [P §5]'s point that slack variables make QUBO models harder.

## What this adds to the paper's future-work list

- **Capacitated location fits the QUBO/QBA framework,** and is verified exactly on [UL]'s example.
- **For a classical single-flip solver, the formulation choices that look equivalent are not:**
  - how the capacity row is coupled to y matters more than P;
  - the slack's binary encoding decides whether raw reads are feasible.
- **Testable next step.** Unary or "one-hot" slack encodings, or a solver with multi-flip moves,
  should reduce the slack barrier. Whether Gurobi's QUBO solver or D-Wave's hybrid solver is
  sensitive to the same choices is an open question; these results say nothing about them.

## Limitations

- **SA at one setting only,** 3 seeds in the paired comparison, and 12 cost matrices.
- **Capacity by customer count (A.4);** demand-based capacity was verified only on [UL]'s small
  example.
- **The repair is a greedy assignment by largest demand first,** not an exact assignment.
