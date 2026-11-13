"""Fresh test set for the penalty model: UflLib's Bilde-Krarup instances (B, C, Dq/1, Eq/1; 40).

Pre-registered on 13 Nov 2026, before any result on these instances was seen: they are used only
for testing; every model is trained on the 250 synthetic instances; the penalty grid, seeds and SA
settings are those of the synthetic labels (scripts/make_labels.py).

Stage "instances": parse each file, solve it exactly (HiGHS), check that the published optimal
assignment (.opt) has exactly that cost, and compute the LP bound and features.
    -> results/bk_instances_<ts>.csv
Stage "runs": SA at P = r x LP bound on the label grid, 3 seeds.  -> results/bk_runs_<ts>.csv (+ .json)

Run:  python scripts/run_bk.py instances
      python scripts/run_bk.py runs results/bk_instances_<ts>.csv [--workers 6]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_labels import READS, R_GRID, SEEDS, SWEEPS  # noqa: E402

from uflp.experiment import CsvLog, run_one  # noqa: E402
from uflp.features import features  # noqa: E402
from uflp.io import DATA_DIR, Instance, parse_ufllib_simple, ufllib_opt_assignment  # noqa: E402
from uflp.milp import lp_bound, solution_cost, solve_uflp  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
BK = DATA_DIR / "raw" / "bk"
NAMES = [f"{s}1.{k}" for s in "BCDE" for k in range(1, 11)]


@lru_cache(maxsize=None)
def load_bk(name: str) -> Instance:
    """Parse one Bilde-Krarup file."""
    return parse_ufllib_simple((BK / name).read_text(), name)


def instance_row(name: str) -> dict:
    """Exact solve, check against the published optimal assignment, LP bound, features."""
    inst = load_bk(name)
    t0 = time.perf_counter()
    res = solve_uflp(inst, time_limit=1800)
    milp_s = time.perf_counter() - t0
    assign, pub_cost = ufllib_opt_assignment((BK / f"{name}.opt").read_text(), inst.n)
    pub = solution_cost(inst, sorted(set(assign)), assign)        # cost of the published assignment
    lp = lp_bound(inst)
    return {"instance": name, "family": f"bilde_krarup_{name[0]}", "m": inst.m, "n": inst.n,
            "opt": res.cost, "published_opt_cost": pub, "published_cost_value": pub_cost,
            "milp_status": res.status,
            "milp_time_s": round(milp_s, 2), "n_open_opt": len(res.open_facilities),
            "lp_gap_pct": 100 * (res.cost - lp) / res.cost, **features(inst, lp=lp)}


def run_job(spec: dict) -> dict:
    """One SA run at P = r x LP bound (module-level for worker processes)."""
    return run_one(load_bk(spec["instance"]), spec["family"], spec["instance"], spec["opt"],
                   spec["r"], spec["seed"], READS, SWEEPS, p_base=spec["lp"])


def main() -> int:
    """Dispatch to a stage."""
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["instances", "runs"])
    ap.add_argument("instances_csv", type=Path, nargs="?")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.stage == "instances":
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(instance_row, NAMES))
        bad = [r["instance"] for r in rows
               if abs(r["opt"] - r["published_opt_cost"]) > 1e-6 * r["opt"]
               or (r["published_cost_value"] is not None
                   and abs(r["opt"] - r["published_cost_value"]) > 1e-6 * r["opt"])]
        out = RESULTS / f"bk_instances_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        df = pd.DataFrame(rows)
        print(df.groupby("family")[["lp_gap_pct", "milp_time_s", "n_open_opt"]].agg(["min", "median", "max"]).round(2))
        print(f"MILP optimum == published assignment cost == published value on {len(rows) - len(bad)}/{len(rows)}; mismatches: {bad}")
        print(f"Wrote {out}")
        return 1 if bad else 0
    inst = pd.read_csv(args.instances_csv)
    jobs = [{"instance": r.instance, "family": r.family, "opt": float(r.opt), "lp": float(r.lp),
             "r": rr, "seed": s} for r in inst.itertuples() for rr in R_GRID for s in SEEDS]
    random.Random(2026).shuffle(jobs)
    log = CsvLog.new(RESULTS, "bk_runs")
    log.path.with_suffix(".json").write_text(json.dumps(
        {"instances_csv": args.instances_csv.name, "r_grid": R_GRID, "seeds": SEEDS, "reads": READS,
         "sweeps": SWEEPS, "penalty": "P = r * LP bound", "n_jobs": len(jobs)}, indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, fut in enumerate(as_completed([ex.submit(run_job, j) for j in jobs]), 1):
            log.write(fut.result())
            if k % 100 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} done", flush=True)
    print(f"Wrote {log.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
