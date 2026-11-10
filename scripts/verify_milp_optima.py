"""M1 acceptance: solve all 37 instances exactly and compare with the published uncapacitated optima.

Also records the LP-relaxation bound and flags instances whose (f, c) data duplicate an
earlier file (the capacitated-set files differ from cap7x/10x/13x only in capacity,
which the uncapacitated problem ignores).

Writes results/m1_milp_check_<timestamp>.csv. Run:  python scripts/verify_milp_optima.py
"""
from __future__ import annotations

import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

from uflp.io import load_instance, load_optima
from uflp.milp import lp_bound, solve_uflp

RESULTS = Path(__file__).resolve().parents[1] / "results"
REL_TOL = 1e-6


def data_fingerprint(inst) -> str:
    """Hash of (f, c) only, so files that differ just in capacity/demand collide."""
    return hashlib.sha1(inst.f.tobytes() + inst.c.tobytes()).hexdigest()[:10]


def main() -> int:
    """Solve every instance, compare with the published optimum, write one CSV row each."""
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"m1_milp_check_{datetime.now():%Y%m%d_%H%M%S}.csv"
    seen: dict[str, str] = {}
    rows, failures = [], 0
    for r in load_optima():
        inst = load_instance(r["name"])
        res = solve_uflp(inst)
        lp = lp_bound(inst)
        rel = abs(res.cost - r["opt"]) / r["opt"]
        ok = rel <= REL_TOL
        failures += not ok
        fp = data_fingerprint(inst)
        dup = seen.setdefault(fp, r["name"])
        rows.append({
            "instance": r["name"], "class": r["class"], "m": inst.m, "n": inst.n,
            "published_opt": r["opt"], "opt_source": r["source"], "milp_cost": res.cost,
            "rel_diff": rel, "match": ok, "lp_bound": lp, "lp_gap_pct": 100 * (res.cost - lp) / res.cost,
            "n_open": len(res.open_facilities), "time_s": round(res.time_s, 3),
            "same_data_as": "" if dup == r["name"] else dup,
        })
        print(f"{r['name']:7s} {r['class']}  milp={res.cost:<14.4f} pub={r['opt']:<14} "
              f"rel={rel:.1e} {'OK ' if ok else 'BAD'} lp_gap={rows[-1]['lp_gap_pct']:.3f}% "
              f"open={len(res.open_facilities):2d} t={res.time_s:.2f}s"
              + (f"  (same f,c as {dup})" if dup != r["name"] else ""))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows) - failures}/{len(rows)} match within rel tol {REL_TOL}; "
          f"{len(set(seen))} distinct (f, c) instances. Wrote {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
