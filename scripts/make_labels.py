"""M4 label data for the ML extension (CLAUDE.md §11, decisions of 11 Nov 2026).

Stage "instances": for each of the 200 seeded specs from `generate.design()`, build the instance
and record its exact optimum (labels/evaluation only, never a feature), the MILP status and time,
and every feature. -> results/ml_instances_<ts>.csv

Stage "runs": SA on every instance at P = r x LP bound for the 12-value grid, 3 seeds, 50 reads x
1,000 sweeps (same SA settings as M3). -> results/ml_runs_<ts>.csv (+ .json settings), resumable.
The label r* (best mean repaired cost per r; ties: lower CPU time) is derived from these rows at M5.

Run:     python scripts/make_labels.py instances [--design heavy]
         python scripts/make_labels.py runs results/ml_instances_<ts>.csv [--workers 6]
Resume:  python scripts/make_labels.py runs results/ml_instances_<ts>.csv --resume results/ml_runs_<ts>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from uflp.experiment import CsvLog, run_one
from uflp.features import features
from uflp.generate import design, design_heavy, generate
from uflp.io import Instance
from uflp.milp import lp_bound, solve_uflp

RESULTS = Path(__file__).resolve().parents[1] / "results"
R_GRID = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.25, 0.5, 1.0]
SEEDS = [0, 1, 2]
READS, SWEEPS = 50, 1000
MILP_TIME_LIMIT = 600.0


def instance_row(spec: dict) -> dict:
    """Generate one instance; solve it exactly and compute its LP bound and features."""
    inst = generate(spec["family"], spec["m"], spec["n"], spec["phi"], spec["seed"])
    t0 = time.perf_counter()
    res = solve_uflp(inst, time_limit=MILP_TIME_LIMIT)
    milp_s = time.perf_counter() - t0
    lp = lp_bound(inst)
    return {"instance": inst.name, **spec, "opt": res.cost, "milp_status": res.status,
            "milp_time_s": round(milp_s, 3), "n_open_opt": len(res.open_facilities),
            "lp_gap_pct": 100 * (res.cost - lp) / res.cost, **features(inst, lp=lp)}


@lru_cache(maxsize=None)
def cached_instance(family: str, m: int, n: int, phi: float, seed: int) -> Instance:
    """Regenerate each instance once per worker process."""
    return generate(family, m, n, phi, seed)


def run_job(spec: dict) -> dict:
    """One SA run at P = r x LP bound (module-level for worker processes)."""
    inst = cached_instance(spec["family"], spec["m"], spec["n"], spec["phi"], spec["gen_seed"])
    return run_one(inst, spec["family"], spec["instance"], spec["opt"], spec["r"], spec["seed"],
                   READS, SWEEPS, p_base=spec["lp"])


def stage_instances(workers: int, which: str) -> None:
    """Solve and featurize every instance of a design in parallel; write them in design order."""
    specs = design() if which == "base" else design_heavy()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(instance_row, specs))
    bad = [r["instance"] for r in rows if r["milp_status"] != "Optimal"]
    if bad:
        raise SystemExit(f"MILP not optimal for {bad}; raise MILP_TIME_LIMIT")
    prefix = "ml_instances" if which == "base" else f"ml_instances_{which}"
    out = RESULTS / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    df = pd.DataFrame(rows)
    print(df.groupby("family")[["lp_gap_pct", "milp_time_s", "n_open_opt"]].describe().round(2).T)
    print(f"Wrote {out}")


def stage_runs(instances_csv: Path, workers: int, resume: Path | None) -> None:
    """Run the SA grid on every instance; append one row per run."""
    inst = pd.read_csv(instances_csv)
    jobs = [{"instance": r.instance, "family": r.family, "m": int(r.m), "n": int(r.n),
             "phi": float(r.phi), "gen_seed": int(r.seed), "opt": float(r.opt), "lp": float(r.lp),
             "r": rr, "seed": s}
            for r in inst.itertuples() for rr in R_GRID for s in SEEDS]
    random.Random(2026).shuffle(jobs)
    if resume:
        log = CsvLog(resume)
        done = pd.read_csv(resume)
        keys = set(zip(done["instance"], done["r"], done["seed"]))
        jobs = [j for j in jobs if (j["instance"], j["r"], j["seed"]) not in keys]
        print(f"resuming {resume}: {len(keys)} done, {len(jobs)} to go")
    else:
        log = CsvLog.new(RESULTS, "ml_runs")
        log.path.with_suffix(".json").write_text(json.dumps({
            "instances_csv": instances_csv.name, "r_grid": R_GRID, "seeds": SEEDS, "reads": READS,
            "sweeps": SWEEPS, "penalty": "P = r * LP bound", "n_jobs": len(jobs), "workers": workers,
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: version(p) for p in ["dimod", "dwave-samplers", "numpy", "pulp", "highspy"]},
        }, indent=2))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_job, j) for j in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            log.write(fut.result())
            if k % 100 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} done", flush=True)
    print(f"Wrote {log.path}")


def main() -> None:
    """Dispatch to a stage."""
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["instances", "runs"])
    ap.add_argument("instances_csv", type=Path, nargs="?")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--design", choices=["base", "heavy"], default="base",
                    help="base: the 200 original instances; heavy: the 50 heavy_demand instances (M5)")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()
    if args.stage == "instances":
        stage_instances(args.workers, args.design)
    else:
        if args.instances_csv is None:
            ap.error("runs stage needs the instances CSV")
        stage_runs(args.instances_csv, args.workers, args.resume)


if __name__ == "__main__":
    main()
