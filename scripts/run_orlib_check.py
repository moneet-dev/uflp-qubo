"""M5 ORLIB sanity check: SA on the 12 distinct ORLIB problems at the label-grid penalties that the
M3 sweep did not cover (3, 4, 6, 8 and 15%), with the same settings as M3 (50 reads x 1,000 sweeps,
seeds 0-9). On ORLIB the LP bound equals the optimum, so P = r x LP = r x opt and these rows
combine directly with results/reproduction_*.csv.

Writes results/orlib_extra_<timestamp>.csv (+ .json). Run:  python scripts/run_orlib_check.py
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from uflp.experiment import CsvLog, run_one
from uflp.io import Instance, load_instance, load_optima

RESULTS = Path(__file__).resolve().parents[1] / "results"
PROBLEMS = [f"cap{k}" for k in (71, 72, 73, 74, 101, 102, 103, 104, 131, 132, 133, 134)]
R_EXTRA = [0.03, 0.04, 0.06, 0.08, 0.15]
SEEDS = list(range(10))
READS, SWEEPS = 50, 1000


@lru_cache(maxsize=None)
def instance(name: str) -> Instance:
    """Parse each file once per worker."""
    return load_instance(name)


def job(spec: dict) -> dict:
    """One SA run (module-level for worker processes)."""
    return run_one(instance(spec["name"]), spec["class"], spec["name"], spec["opt"], spec["r"],
                   spec["seed"], READS, SWEEPS)


def main() -> None:
    """Run all extra (problem, r, seed) combinations in parallel."""
    opt = {r["name"]: r for r in load_optima()}
    specs = [{"name": p, "class": opt[p]["class"], "opt": opt[p]["opt"], "r": r, "seed": s}
             for p in PROBLEMS for r in R_EXTRA for s in SEEDS]
    random.Random(2026).shuffle(specs)
    log = CsvLog.new(RESULTS, "orlib_extra")
    log.path.with_suffix(".json").write_text(json.dumps(
        {"problems": PROBLEMS, "r_extra": R_EXTRA, "seeds": SEEDS, "reads": READS,
         "sweeps": SWEEPS, "penalty": "P = r * opt (= r * LP bound on ORLIB)"}, indent=2))
    with ProcessPoolExecutor(max_workers=6) as ex:
        for k, fut in enumerate(as_completed([ex.submit(job, s) for s in specs]), 1):
            log.write(fut.result())
            if k % 100 == 0 or k == len(specs):
                print(f"{k}/{len(specs)} done", flush=True)
    print(f"Wrote {log.path}")


if __name__ == "__main__":
    main()
