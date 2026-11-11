"""M3 reproduction sweep: P = r * opt for r in the CLAUDE.md §10 grid, 10 seeds, all 37 files.

The grid contains the paper's four values (0.25, 0.5, 0.75, 1.0) [P §4.5] plus smaller ones
for single-flip SA. Penalty-function QUBO (identical model to the Q-matrix one, verified at M2),
50 reads x 1,000 sweeps (chosen in results/sweeps_check_*.csv: 10x sweeps gained < 2 points).

Jobs run in worker processes in a fixed shuffled order, so parallel load is spread evenly over r;
time_s is wall time under that load, cpu_time_s is unaffected by it. The main process is the
only writer of the CSV (one row per run, flushed after each), so the file is always valid.
A JSON file next to the CSV records the settings and package versions.

Run:      python scripts/run_reproduction.py [--workers 6]
Resume:   python scripts/run_reproduction.py --resume results/reproduction_<ts>.csv
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from uflp.experiment import CsvLog, fingerprint, run_one
from uflp.io import Instance, load_instance, load_optima

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
R_GRID = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
SEEDS = list(range(10))
READS, SWEEPS, BUILDER = 50, 1000, "penalty"
CANONICAL = {"cap7", "cap10", "cap13"}   # the files from the uncapacitated set name each problem


@lru_cache(maxsize=None)
def instance(name: str) -> Instance:
    """Parse each file once per worker process."""
    return load_instance(name)


def job(spec: dict) -> dict:
    """One run; module-level so it can be sent to worker processes."""
    return run_one(instance(spec["instance"]), spec["class"], spec["problem"], spec["opt"],
                   spec["r"], spec["seed"], READS, SWEEPS, builder=BUILDER)


def problem_names(rows: list[dict]) -> dict[str, str]:
    """Map each file to the name of its distinct (f, c) problem, preferring cap7x/10x/13x."""
    by_fp: dict[str, list[str]] = {}
    for r in rows:
        by_fp.setdefault(fingerprint(instance(r["name"])), []).append(r["name"])
    out = {}
    for names in by_fp.values():
        canon = next((n for n in names if n[:-1] in CANONICAL), names[0])   # cap71 -> "cap7"
        for n in names:
            out[n] = canon
    return out


def write_metadata(csv_path: Path, n_jobs: int, workers: int) -> None:
    """Record settings, versions and git commit next to the results CSV."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    meta = {
        "r_grid": R_GRID, "seeds": SEEDS, "reads": READS, "sweeps": SWEEPS, "builder": BUILDER,
        "penalty": "P = r * published optimum", "n_jobs": n_jobs, "workers": workers,
        "git_commit_at_start": commit, "python": platform.python_version(),
        "platform": platform.platform(), "cpu_count_logical": __import__("os").cpu_count(),
        "packages": {p: version(p) for p in ["dimod", "dwave-samplers", "numpy", "pulp", "highspy"]},
    }
    csv_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    """Build the job list, skip finished runs when resuming, run the rest in parallel."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    rows = load_optima()
    canon = problem_names(rows)
    specs = [{"instance": r["name"], "class": r["class"], "problem": canon[r["name"]],
              "opt": r["opt"], "r": rr, "seed": s}
             for r in rows for rr in R_GRID for s in SEEDS]
    random.Random(2026).shuffle(specs)

    if args.resume:
        log = CsvLog(args.resume)
        done = pd.read_csv(args.resume)
        keys = set(zip(done["instance"], done["r"], done["seed"]))
        specs = [s for s in specs if (s["instance"], s["r"], s["seed"]) not in keys]
        print(f"resuming {log.path}: {len(keys)} done, {len(specs)} to go")
    else:
        log = CsvLog.new(RESULTS, "reproduction")
        write_metadata(log.path, len(specs), args.workers)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(job, s) for s in specs]
        for k, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            log.write(row)
            if k % 50 == 0 or k == len(specs):
                print(f"{k}/{len(specs)} done", flush=True)
    print(f"Wrote {log.path}")


if __name__ == "__main__":
    main()
