"""M4 label-data checks: is the synthetic SA sweep complete, consistent and reproducible?

Checks
  1. instances     200 instances, all solved to proven optimality, LP <= opt, features finite
  2. completeness  every (instance, r, seed) of the grid appears exactly once
  3. bookkeeping   p_base = LP bound of the instance, P = r * p_base, gap recomputes vs opt
  4. invariants    no repaired or raw-feasible cost below the optimum; when P exceeds the
                   sufficient exactness bound, no energy below the optimum
  5. reproducible  a sample of runs re-executed serially gives identical results

Writes results/ml_verification_<timestamp>.txt and prints the label distribution.
Run:  python scripts/verify_labels.py results/ml_instances_<ts>.csv results/ml_runs_<ts>.csv
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from uflp.experiment import run_one
from uflp.generate import generate

FIELDS = ["best_energy", "raw_feasible", "n_feasible_reads", "n_no_facility_reads",
          "raw_best_cost", "repaired_cost"]


def same(a, b) -> bool:
    """Exact equality treating NaN == NaN and numpy/bool/int alike."""
    a = a.item() if hasattr(a, "item") else a
    b = b.item() if hasattr(b, "item") else b
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return float(a) == float(b) if not isinstance(a, bool) else bool(a) == bool(b)


def labels(runs: pd.DataFrame) -> pd.DataFrame:
    """r* per instance: lowest mean repaired cost over seeds; ties broken by lower mean CPU time.

    A run whose reads all opened no facility has no solution (repaired_cost NaN); it counts as
    infinite cost, so a penalty that sometimes fails can never win on the runs that succeeded.
    (On the 11 Nov data this changes no label: such runs occur only at r <= 0.02.)
    """
    runs = runs.assign(repaired_cost=runs["repaired_cost"].fillna(np.inf))
    g = runs.groupby(["instance", "r"]).agg(cost=("repaired_cost", "mean"),
                                           cpu=("cpu_time_s", "mean")).reset_index()
    g = g.sort_values(["instance", "cost", "cpu"])
    return g.groupby("instance").head(1).set_index("instance")


def main() -> int:
    """Run the checks, write the report, print label statistics."""
    ap = argparse.ArgumentParser()
    ap.add_argument("instances", type=Path)
    ap.add_argument("runs", type=Path)
    ap.add_argument("--sample", type=int, default=12)
    args = ap.parse_args()
    inst = pd.read_csv(args.instances).set_index("instance")
    runs = pd.read_csv(args.runs)
    meta = json.loads(args.runs.with_suffix(".json").read_text())
    lines, ok_all = [], True

    def report(name: str, ok: bool, detail: str) -> None:
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        print(lines[-1], flush=True)

    feat_cols = [c for c in inst.columns if c not in ("family", "milp_status")]
    report("instances", len(inst) > 0 and (inst["milp_status"] == "Optimal").all()
           and (inst["lp"] <= inst["opt"] * (1 + 1e-9)).all()
           and np.isfinite(inst[feat_cols].to_numpy(dtype=float)).all(),
           f"{len(inst)} instances, all optimal {(inst['milp_status'] == 'Optimal').all()}, "
           f"LP <= opt {(inst['lp'] <= inst['opt'] * (1 + 1e-9)).all()}")

    expect = {(i, r, s) for i in inst.index for r in meta["r_grid"] for s in meta["seeds"]}
    got = list(zip(runs["instance"], runs["r"], runs["seed"]))
    report("completeness", set(got) == expect and len(got) == len(expect),
           f"{len(got)} rows, {len(set(got))} unique, {len(expect)} expected")

    lp = runs["instance"].map(inst["lp"])
    opt = runs["instance"].map(inst["opt"])
    gap = 100 * (runs["repaired_cost"] - opt) / opt
    book = ((runs["p_base"] == lp).all() and (abs(runs["P"] - runs["r"] * runs["p_base"]) <= 1e-9 * runs["P"]).all()
            and (runs["opt"] == opt).all() and ((gap - runs["gap_pct"]).abs().fillna(0) < 1e-9).all())
    report("bookkeeping", book, "p_base = LP bound, P = r * p_base, opt from instances file, gap recomputes")

    tol = 1e-9 * opt
    rep_ok = (runs["repaired_cost"].dropna() >= (opt - tol)[runs["repaired_cost"].notna()]).all()
    raw_ok = (runs["raw_best_cost"].dropna() >= (opt - tol)[runs["raw_best_cost"].notna()]).all()
    exact = runs[runs["P"] > runs["instance"].map(inst["bound_suf"])]
    en_ok = (exact["best_energy"] >= exact["opt"] * (1 - 1e-9)).all()
    report("invariants", rep_ok and raw_ok and en_ok,
           f"repaired>=opt {rep_ok}, raw>=opt {raw_ok}, energy>=opt where P exact ({len(exact)} runs) {en_ok}")

    rng = random.Random(11)
    sample = rng.sample(runs.to_dict("records"), args.sample)
    mism = 0
    for x in sample:
        s = inst.loc[x["instance"]]
        i = generate(s["family"], int(s["m"]), int(s["n"]), float(s["phi"]), int(s["seed"]))
        again = run_one(i, x["class"], x["instance"], x["opt"], x["r"], int(x["seed"]),
                        int(x["reads"]), int(x["sweeps"]), p_base=x["p_base"])
        mism += not all(same(x[f], again[f]) for f in FIELDS)
    report("reproducible", mism == 0, f"{len(sample)} sampled runs re-executed, {mism} differ")

    lab = labels(runs).join(inst[["family", "lp_gap_pct"]])
    lines.append("\nLabel r* (best r by mean repaired cost) per family, counts:")
    lines.append(pd.crosstab(lab["family"], lab["r"]).to_string())
    lines.append(f"\nmedian r* per family: {lab.groupby('family')['r'].median().to_dict()}")
    print("\n".join(lines[-3:]))
    out = args.runs.parent / f"ml_verification_{datetime.now():%Y%m%d_%H%M%S}.txt"
    out.write_text(f"Verification of {args.runs.name}\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{'ALL CHECKS PASS' if ok_all else 'SOME CHECKS FAILED'}; wrote {out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
