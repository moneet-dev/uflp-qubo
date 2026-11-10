"""M2 pilot (CLAUDE.md §13): does the synthetic pilot's finding (best P ~ 2-5% of opt for SA)
hold on ORLIB A-1? Runs the M3 r-grid on the 4 distinct A-1 problems (cap71-74) with a few
seeds, before committing to the full 2,960-run sweep.

Writes results/sa_pilot_<timestamp>.csv (one row per run) and prints a summary per r.
Run:  python scripts/run_sa_pilot.py [--seeds 3] [--reads 50] [--sweeps 1000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from uflp.experiment import CsvLog, run_one
from uflp.io import load_instance, load_optima

RESULTS = Path(__file__).resolve().parents[1] / "results"
R_GRID = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
PROBLEMS = ["cap71", "cap72", "cap73", "cap74"]   # the 4 distinct A-1 problems (§7.1)


def main() -> None:
    """Run the pilot grid and print mean gaps and feasibility per r."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reads", type=int, default=50)
    ap.add_argument("--sweeps", type=int, default=1000)
    args = ap.parse_args()
    opt = {r["name"]: r for r in load_optima()}
    log = CsvLog.new(RESULTS, "sa_pilot")
    for name in PROBLEMS:
        inst = load_instance(name)
        for r in R_GRID:
            for seed in range(args.seeds):
                row = run_one(inst, opt[name]["class"], name, opt[name]["opt"], r, seed,
                              args.reads, args.sweeps)
                log.write(row)
                print(f"{name} r={r:<5} seed={seed} feas_reads={row['n_feasible_reads']:3d}/"
                      f"{args.reads} gap={row['gap_pct']:7.3f}% raw_gap={row['raw_gap_pct']:7.3f}% "
                      f"t={row['time_s']:.2f}s", flush=True)
    df = pd.read_csv(log.path)
    df["feas_rate"] = df["n_feasible_reads"] / df["reads"]
    summary = df.groupby("r").agg(
        mean_gap_pct=("gap_pct", "mean"), max_gap_pct=("gap_pct", "max"),
        mean_raw_gap_pct=("raw_gap_pct", "mean"), feas_rate=("feas_rate", "mean"),
        no_facility_reads=("n_no_facility_reads", "sum"), mean_time_s=("time_s", "mean"))
    print("\n" + summary.round(3).to_string())
    print(f"\nWrote {log.path}")


if __name__ == "__main__":
    main()
