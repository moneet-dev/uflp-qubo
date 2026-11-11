"""Capacitated extension: does the penalty behave the same once capacity constraints (and their
slack variables) are in the QUBO?

Problems: the 12 distinct ORLIB problems (cap7x, cap10x, cap13x costs) with the capacity of
Li (2026), Urban Lifeline, Appendix A (A.4): each facility serves at most k customers (w = 1,
Q = k), k in {5, 10}. ORLIB's own demand capacities cannot be used with single sourcing: one
customer's demand (12,912) exceeds most facilities' capacity (5,000 or 10,000).

Each problem is solved exactly (single-source CFLP MILP, HiGHS), then by SA on the QUBO with
P = r x CFLP optimum, r on the M3 grid, 5 seeds, 50 reads x 1,000 sweeps (same as M3).
Writes results/capacitated[_decoupled]_<ts>.csv (+ .json).
Run:  python scripts/run_capacitated.py [--formulation decoupled] [--r-grid ...] [--seeds N]
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np

from uflp.capacitated import CapInstance, build_cflp_qubo, run_cflp_sa, solve_cflp
from uflp.experiment import gap
from uflp.io import load_instance, load_optima

RESULTS = Path(__file__).resolve().parents[1] / "results"
PROBLEMS = [f"cap{k}" for k in (71, 72, 73, 74, 101, 102, 103, 104, 131, 132, 133, 134)]
KS = [5, 10]
R_GRID = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
SEEDS = list(range(5))
READS, SWEEPS = 50, 1000
COLUMNS = ["instance", "problem", "class", "k", "m", "n", "opt", "opt_uflp", "binding", "r", "P",
           "seed", "reads", "sweeps", "n_vars", "best_energy", "n_feasible_reads",
           "n_slack_exact_reads", "raw_best_cost", "raw_gap_pct", "repaired_cost", "gap_pct",
           "n_repair_fail", "time_s", "cpu_time_s", "formulation"]


@lru_cache(maxsize=None)
def cap_instance(problem: str, k: int) -> CapInstance:
    """ORLIB costs with at most k customers per facility."""
    u = load_instance(problem)
    return CapInstance(f"{problem}_k{k}", u.f, u.c, np.ones(u.n), np.full(u.m, float(k)))


def job(spec: dict) -> dict:
    """One SA run on one capacitated problem (module-level for worker processes)."""
    inst = cap_instance(spec["problem"], spec["k"])
    p = spec["r"] * spec["opt"]
    form = spec.get("formulation") or "coupled"
    res = run_cflp_sa(inst, build_cflp_qubo(inst, p, couple_y=(form == "coupled")), READS, SWEEPS, spec["seed"])
    row = {**{c: spec.get(c) for c in ["instance", "problem", "class", "k", "opt", "opt_uflp",
                                         "binding", "r", "seed"]},
           "m": inst.m, "n": inst.n, "P": p, "reads": READS, "sweeps": SWEEPS, "formulation": form, **res}
    row["raw_gap_pct"] = gap(row["raw_best_cost"], row["opt"])
    row["gap_pct"] = gap(row["repaired_cost"], row["opt"])
    return {c: row[c] for c in COLUMNS}


def main() -> None:
    """Solve every problem exactly, then run the SA grid in parallel."""
    import argparse
    import csv
    from datetime import datetime
    ap = argparse.ArgumentParser()
    ap.add_argument("--formulation", choices=["coupled", "decoupled"], default="coupled",
                    help="capacity row with Q_i*y_i ([UL] A.4 as written) or with Q_i (see capacitated.py)")
    ap.add_argument("--r-grid", type=float, nargs="+", default=R_GRID)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    args = ap.parse_args()
    r_grid, seeds = args.r_grid, list(range(args.seeds))
    uopt = {r["name"]: r for r in load_optima()}
    probs = []
    for p in PROBLEMS:
        for k in KS:
            res = solve_cflp(cap_instance(p, k), time_limit=600)
            probs.append({"instance": f"{p}_k{k}", "problem": p, "class": uopt[p]["class"], "k": k,
                          "opt": res.cost, "opt_uflp": uopt[p]["opt"],
                          "binding": bool(res.cost > uopt[p]["opt"] * (1 + 1e-9))})
            print(f"{p} k={k}: CFLP opt {res.cost:.2f} (uncap {uopt[p]['opt']:.2f})", flush=True)
    specs = [{**pr, "r": r, "seed": s, "formulation": args.formulation}
             for pr in probs for r in r_grid for s in seeds]
    random.Random(2026).shuffle(specs)
    tag = "capacitated" if args.formulation == "coupled" else "capacitated_decoupled"
    out = RESULTS / f"{tag}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.with_suffix(".json").write_text(json.dumps(
        {"problems": PROBLEMS, "k": KS, "r_grid": r_grid, "seeds": seeds, "reads": READS,
         "formulation": args.formulation,
         "sweeps": SWEEPS, "capacity": "Urban Lifeline (A.4): at most k customers per facility",
         "penalty": "P = r * CFLP optimum; capacity rows scaled by mean weight (1 here)",
         "optima": {p["instance"]: p["opt"] for p in probs}}, indent=2))
    with open(out, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=COLUMNS).writeheader()
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, fut in enumerate(as_completed([ex.submit(job, s) for s in specs]), 1):
            with open(out, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=COLUMNS).writerow(fut.result())
            if i % 50 == 0 or i == len(specs):
                print(f"{i}/{len(specs)} done", flush=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
