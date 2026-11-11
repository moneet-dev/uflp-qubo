"""Choose the SA effort for M3: repaired and raw gap vs number of sweeps on cap74 (the hardest A-1 problem).

Writes results/sweeps_check_<timestamp>.csv. Run:  python scripts/check_sweeps.py
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from uflp.experiment import CsvLog, run_one
from uflp.io import load_instance, load_optima

RESULTS = Path(__file__).resolve().parents[1] / "results"
NAME, READS, SEEDS = "cap74", 50, 3
SWEEPS = [1000, 3000, 10000]
R_GRID = [0.05, 0.1, 0.25, 0.5]


def job(args: tuple[float, int, int]) -> dict:
    """One SA run on cap74 (module-level so worker processes can pickle it)."""
    r, sweeps, seed = args
    row = next(x for x in load_optima() if x["name"] == NAME)
    return run_one(load_instance(NAME), row["class"], NAME, row["opt"], r, seed, READS, sweeps)


def main() -> None:
    """Run the grid in parallel and print mean gaps and time per (sweeps, r)."""
    log = CsvLog.new(RESULTS, "sweeps_check")
    jobs = [(r, s, seed) for s in SWEEPS for r in R_GRID for seed in range(SEEDS)]
    with ProcessPoolExecutor(max_workers=6) as ex:
        for row in ex.map(job, jobs):
            log.write(row)
    df = pd.read_csv(log.path)
    print(df.pivot_table(index="r", columns="sweeps", values="gap_pct", aggfunc="mean").round(2))
    print(df.pivot_table(index="r", columns="sweeps", values="raw_gap_pct", aggfunc="mean").round(2))
    print(df.groupby("sweeps")["time_s"].mean().round(2))
    print(f"Wrote {log.path}")


if __name__ == "__main__":
    main()
