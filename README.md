# QUBO penalty study for the uncapacitated facility location problem

A small, reproducible computational study built on

> Li, H., Hennig, R., Kochenberger, G., & Glover, F. (2026). Quantum bridge analytics: a QUBO
> approach to the uncapacitated facility location problem. *Annals of Operations Research*.
> https://doi.org/10.1007/s10479-026-07201-1

The paper reformulates the uncapacitated facility location problem (UFLP) as a QUBO by moving its
constraints into the objective with a penalty P. It found that P = 25–100% of the optimal cost works
when the QUBO is solved with Gurobi's QUBO solver.

**Question 1:** does the choice of P behave the same when the QUBO is solved by classical
**simulated annealing (SA)**?
**Question 2 (in progress):** can a model predict a good P for a new instance from cheap features,
without knowing the optimal cost?

All results here are from classical simulated annealing. They are not quantum results and not
Gurobi results.

## Main findings so far

On the paper's OR-Library instances (classes A-1, A-2, B-1: 37 files = 12 distinct problems), with
2,960 seeded SA runs (full write-up: [`report/m3_summary.md`](report/m3_summary.md)):

| | A-1 | A-2 | B-1 |
|---|---|---|---|
| SA gap at the paper's P = 50% (raw, as measured in the paper) | 56.6% | 80.1% | 109.6% |
| SA gap at P = 50% after repair | 5.6% | 10.9% | 28.3% |
| SA gap at its best P = 5%, after repair | **4.0%** | **7.7%** | **13.9%** |
| Paper, Gurobi QUBO solver, P = 25–100% | 0% | 0% | 0% |

1. **The best penalty depends on the solver.** For single-flip SA it is about 5% of the optimum, well
   below the paper's range. Across the paper's 25–100%, SA gets steadily worse as P grows.
2. **The paper's "no infeasibility at 25%" has a simple explanation.** The QUBO minimum is guaranteed
   feasible once P > max(max_j min_i (f_i + c_ij), max_i f_i). That is 16.6–22.6% of the optimum on these
   instances, so every QUBO in the paper's range is exact (`uflp.qubo.penalty_bounds`).
3. **More SA effort does not close the gap** (10× sweeps gain < 2 points), which is consistent with
   energy barriers of size about P.

![SA gap after repair vs penalty](results/figures/m3_gap_repaired.png)

## Reproduce

Requires Python 3.12 or 3.13.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -e .   # on Linux/macOS: .venv/bin/python
python data/download.py                    # 37 ORLIB files + uncapacitated optima (checked)
pytest -q                                  # 99 tests, including the paper's worked example
python scripts/verify_milp_optima.py       # M1: exact MILP matches all 37 published optima
python scripts/run_reproduction.py         # M3: 2,960 SA runs (about 1 h on 6 cores)
python scripts/verify_m3_results.py results/reproduction_<timestamp>.csv
python scripts/make_figures.py results/reproduction_<timestamp>.csv
```

Every run is seeded; results go to timestamped files in `results/` and are never overwritten.

## Layout

| Path | What it is |
|---|---|
| `data/download.py` | downloads instances and optima, with consistency checks |
| `src/uflp/io.py` | OR-Library parser (costs are stored per customer, so it transposes them) |
| `src/uflp/milp.py` | exact solve and LP relaxation (PuLP + HiGHS) |
| `src/uflp/qubo.py` | the paper's two QUBO constructions (eq. 15 and Properties 1–2), exactness bounds |
| `src/uflp/anneal.py`, `repair.py` | seeded SA runner; repair to a feasible solution |
| `src/uflp/experiment.py` | one run → one CSV row |
| `scripts/` | the M1–M3 pipelines, checks and figures |
| `results/`, `report/` | outputs and write-ups |

## Differences from the paper

- Simulated annealing (`dwave-samplers`, 50 reads × 1,000 sweeps) instead of Gurobi's QUBO solver,
  D-Wave's hybrid solver or NGQ.
- Classes A-1, A-2 and B-1 only.
- Quality is reported both raw (as in the paper) and after a repair step that the paper did not need.
- Timing is SA CPU time, not the paper's time to the best solution, so the two are not directly comparable.

## Data sources

- OR-Library (J. E. Beasley): https://people.brunel.ac.uk/~mastjjb/jeb/orlib/uncapinfo.html
- Uncapacitated optima for the capacitated-set files: the UflLib ORLIB package (M. Hoefer, MPI
  Informatik). UflLib was unreachable in November 2026, so a pinned GitHub copy is used and checked
  three ways (see `data/download.py`).
