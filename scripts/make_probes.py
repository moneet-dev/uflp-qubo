"""Compute probe features (src/uflp/probe.py) for every instance used by the penalty model:
the synthetic instances, the 12 distinct ORLIB problems and the 40 Bilde-Krarup test instances.

Each instance's LP bound and greedy cost are read from its instances CSV (already computed).
Writes results/probe_features_<ts>.csv. Run:
    python scripts/make_probes.py --synthetic results/ml_instances_<ts>.csv [more ...] \
        --bk results/bk_instances_<ts>.csv
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bk import load_bk  # noqa: E402

from uflp.features import features  # noqa: E402
from uflp.generate import generate  # noqa: E402
from uflp.io import load_instance  # noqa: E402
from uflp.milp import lp_bound  # noqa: E402
from uflp.probe import PROBE_FEATURES, probe_features  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
ORLIB = [f"cap{k}" for k in (71, 72, 73, 74, 101, 102, 103, 104, 131, 132, 133, 134)]


def job(spec: dict) -> dict:
    """Probe one instance (module-level for worker processes)."""
    if spec["source"] == "synthetic":
        inst = generate(spec["family"], spec["m"], spec["n"], spec["phi"], spec["seed"])
    elif spec["source"] == "orlib":
        inst = load_instance(spec["instance"])
    else:
        inst = load_bk(spec["instance"])
    lp, greedy = spec["lp"], spec["greedy"]
    if lp is None:                                   # ORLIB: compute LP and greedy here
        ft = features(inst, lp=lp_bound(inst))
        lp, greedy = ft["lp"], ft["greedy"]
    return {"instance": spec["instance"], "source": spec["source"], **probe_features(inst, lp, greedy)}


def main() -> None:
    """Probe all instances in parallel."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=Path, nargs="+", required=True)
    ap.add_argument("--bk", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    specs = []
    for f in args.synthetic:
        for r in pd.read_csv(f).itertuples():
            specs.append({"source": "synthetic", "instance": r.instance, "family": r.family, "m": int(r.m),
                          "n": int(r.n), "phi": float(r.phi), "seed": int(r.seed), "lp": float(r.lp),
                          "greedy": float(r.greedy)})
    specs += [{"source": "orlib", "instance": p, "lp": None, "greedy": None} for p in ORLIB]
    for r in pd.read_csv(args.bk).itertuples():
        specs.append({"source": "bk", "instance": r.instance, "lp": float(r.lp), "greedy": float(r.greedy)})
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(job, specs))
    out = RESULTS / f"probe_features_{datetime.now():%Y%m%d_%H%M%S}.csv"
    pd.DataFrame(rows)[["instance", "source"] + PROBE_FEATURES].to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
