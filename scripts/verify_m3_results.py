"""M3 result checks: is the sweep complete, internally consistent, reproducible, and comparable to [P]?

Checks
  1. completeness   every (file, r, seed) of the grid appears exactly once
  2. bookkeeping    P = r*opt, m/n per class, gap_pct recomputes from repaired_cost
  3. invariants     no cost below the optimum; when P exceeds the sufficient exactness
                    bound (qubo.penalty_bounds) the QUBO minimum is opt, so no energy < opt
  4. duplicates     files with identical (f, c) give identical results for the same r and seed
                    (same QUBO, same seed) -> also re-checks the parser on capacitated-format files
  5. reproducible   a sample of runs re-executed serially gives identical results
  6. builders       a sample re-run with the Q-matrix builder gives identical results
  7. paper          raw feasibility at the paper's r values (paper: no infeasibility at all)

Writes results/m3_verification_<timestamp>.txt. Exits non-zero if any check fails.
Run:  python scripts/verify_m3_results.py results/reproduction_<ts>.csv
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from uflp.experiment import run_one
from uflp.io import load_instance, load_optima
from uflp.qubo import penalty_bounds

RESULT_FIELDS = ["best_energy", "raw_feasible", "n_feasible_reads", "n_no_facility_reads",
                 "raw_best_cost", "repaired_cost"]
SIZES = {"A-1": (16, 50), "A-2": (25, 50), "B-1": (50, 50)}
PAPER_R = [0.25, 0.5, 0.75, 1.0]


def same(a, b) -> bool:
    """Exact equality that treats NaN == NaN."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


ENERGY_FIELDS = {"best_energy", "raw_best_cost"}


def rows_equal(a: dict, b: dict, energy_rel_tol: float = 0.0) -> bool:
    """True if two runs agree on every result field (timing excluded).

    energy_rel_tol > 0 lets energies differ by float round-off only: the two QUBO builders
    give coefficients that differ by ~1e-11 (c + P - P vs c), so the same states have
    energies that differ in the last digits, while states, counts and repaired costs match.
    """
    for f in RESULT_FIELDS:
        x, y = _norm(a[f]), _norm(b[f])
        if energy_rel_tol and f in ENERGY_FIELDS and isinstance(x, float) and isinstance(y, float) \
                and not (math.isnan(x) or math.isnan(y)):
            if abs(x - y) > energy_rel_tol * abs(x):
                return False
        elif not same(x, y):
            return False
    return True


def _norm(v):
    """Normalise CSV-read and in-memory values (numpy/bool/str) for comparison."""
    if isinstance(v, str):
        return {"True": True, "False": False}.get(v, v)
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return float(v)
    return v


def main() -> int:
    """Run all checks and write a report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--sample", type=int, default=12)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)   # pandas >= 3 parses floats round-trip exactly (needed for == checks)
    meta = pd.read_json(args.csv.with_suffix(".json"), typ="series")
    opt = {r["name"]: r for r in load_optima()}
    lines, ok_all = [], True

    def report(name: str, ok: bool, detail: str) -> None:
        nonlocal ok_all
        ok_all &= ok
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        print(lines[-1], flush=True)

    # 1. completeness
    expect = {(n, r, s) for n in opt for r in meta["r_grid"] for s in meta["seeds"]}
    got = list(zip(df["instance"], df["r"], df["seed"]))
    report("completeness", set(got) == expect and len(got) == len(expect),
           f"{len(got)} rows, {len(set(got))} unique, {len(expect)} expected")

    # 2. bookkeeping
    p_ok = (abs(df["P"] - df["r"] * df["opt"]) <= 1e-9 * df["P"]).all()
    opt_ok = all(abs(o - opt[n]["opt"]) == 0 for n, o in zip(df["instance"], df["opt"]))
    size_ok = all(SIZES[c] == (m, n) for c, m, n in zip(df["class"], df["m"], df["n"]))
    g = 100 * (df["repaired_cost"] - df["opt"]) / df["opt"]
    gap_ok = ((g - df["gap_pct"]).abs().fillna(0) < 1e-9).all() and (g.isna() == df["gap_pct"].isna()).all()
    report("bookkeeping", bool(p_ok and opt_ok and size_ok and gap_ok),
           f"P=r*opt {p_ok}, opt from optima.csv {opt_ok}, sizes {size_ok}, gap recomputes {gap_ok}")

    # 3. invariants
    tol = 1e-9 * df["opt"]
    rep_ok = (df["repaired_cost"].dropna() >= (df["opt"] - tol)[df["repaired_cost"].notna()]).all()
    raw_ok = (df["raw_best_cost"].dropna() >= (df["opt"] - tol)[df["raw_best_cost"].notna()]).all()
    suf = {n: penalty_bounds(load_instance(n))[1] / opt[n]["opt"] for n in opt}
    exact = df[df["r"] > df["instance"].map(suf)]
    en_ok = (exact["best_energy"] >= exact["opt"] - 1e-9 * exact["opt"]).all()
    below = df[df["best_energy"] < df["opt"] - tol]
    report("invariants", bool(rep_ok and raw_ok and en_ok),
           f"repaired>=opt {rep_ok}, raw>=opt {raw_ok}, energy>=opt where P exact "
           f"({len(exact)} runs) {en_ok}; runs with energy<opt (P below threshold): {len(below)}, "
           f"max r among them {below['r'].max() if len(below) else '-'}")

    # wall time is not trustworthy if the machine slept mid-run; CPU time is unaffected
    slept = df[df["time_s"] > 5 * df["cpu_time_s"] + 60]
    lines.append(f"[INFO] wall time: {len(slept)} runs have wall time > 5x CPU time + 60 s "
                 f"(machine sleep or heavy load; max {df['time_s'].max():.0f} s). "
                 f"Use cpu_time_s for timing; results are unaffected (seeded).")
    print(lines[-1])

    # 4. duplicate files
    mism = 0
    for (_, r, s), grp in df.groupby(["problem", "r", "seed"]):
        recs = grp.to_dict("records")
        mism += sum(not rows_equal(recs[0], x) for x in recs[1:])
    report("duplicates identical", mism == 0, f"{mism} mismatching rows across files of the same problem")

    # 5./6. re-runs (one per problem, random r and seed)
    rng = random.Random(7)
    sample = [rng.choice(g.to_dict("records")) for _, g in df.groupby("problem")][: args.sample]
    rep_mis = bld_mis = 0
    for x in sample:
        inst = load_instance(x["instance"])
        again = run_one(inst, x["class"], x["problem"], x["opt"], x["r"], int(x["seed"]),
                        int(x["reads"]), int(x["sweeps"]), builder="penalty")
        qm = run_one(inst, x["class"], x["problem"], x["opt"], x["r"], int(x["seed"]),
                     int(x["reads"]), int(x["sweeps"]), builder="qmatrix")
        rep_mis += not rows_equal(x, again)
        bld_mis += not rows_equal(x, qm, energy_rel_tol=1e-12)
    report("reproducible", rep_mis == 0, f"{len(sample)} sampled runs re-executed, {rep_mis} differ")
    report("builders agree under SA", bld_mis == 0,
           f"{len(sample)} sampled runs with the Q-matrix builder, {bld_mis} differ "
           f"(states, counts, repaired cost exact; energies to rel 1e-12)")

    # 7. paper comparison (informational for gaps; feasibility is the claim [P] makes)
    pr = df[df["r"].isin(PAPER_R)]
    feas_rate = (pr["n_feasible_reads"] / pr["reads"]).groupby([pr["class"], pr["r"]]).mean()
    lines.append("\nRaw-feasible read rate at the paper's r values ([P]: no infeasibility at any of them):")
    lines.append(feas_rate.unstack().round(4).to_string())
    any_best_infeasible = int((~pr["raw_feasible"].astype(bool)).sum())
    lines.append(f"Runs at the paper's r whose lowest-energy read is infeasible: {any_best_infeasible} of {len(pr)}")
    print("\n".join(lines[-3:]))

    out = args.csv.parent / f"m3_verification_{datetime.now():%Y%m%d_%H%M%S}.txt"
    out.write_text(f"Verification of {args.csv.name}\n\n" + "\n".join(lines) + "\n")
    print(f"\n{'ALL CHECKS PASS' if ok_all else 'SOME CHECKS FAILED'}; wrote {out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
