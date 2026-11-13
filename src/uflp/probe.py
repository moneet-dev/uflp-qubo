"""Probe features: measure how SA itself responds to the penalty on this instance, cheaply.

The cost features describe an instance's data, but not how SA's solution quality changes with P,
and that response is what differs between ORLIB and the synthetic families (M5 follow-up). A probe
runs a tiny SA (5 reads x 200 sweeps, about 2% of one full run) at three penalties and records:

* feas_<r>  share of reads that are feasible
* rep_<r>   best repaired cost relative to the greedy heuristic's cost: repaired/greedy - 1
            (needs no optimum; negative means the probe beat greedy)
* slope_lo  rep_0.05 - rep_0.02   (how much raising P from 2% to 5% helps)
* slope_hi  rep_0.15 - rep_0.05   (how much raising P from 5% to 15% hurts)

Penalties are relative to the LP bound, like the labels. The seed (100) differs from the seeds of
the evaluation runs (0-2). Probes that open no facility score rep = 1 (twice the greedy cost).
This is the "probing" or "landmarking" idea from algorithm-configuration research.
"""
from __future__ import annotations

from .anneal import run_sa
from .io import Instance
from .qubo import build_penalty_function

PROBE_R = (0.02, 0.05, 0.15)
PROBE_READS, PROBE_SWEEPS, PROBE_SEED = 5, 200, 100
PROBE_FEATURES = [f"feas_{r:g}" for r in PROBE_R] + [f"rep_{r:g}" for r in PROBE_R] + ["slope_lo", "slope_hi"]


def probe_features(inst: Instance, lp: float, greedy: float) -> dict:
    """Run the three probes and return the probe features (no optimum used)."""
    out = {}
    for r in PROBE_R:
        res = run_sa(inst, build_penalty_function(inst, r * lp), PROBE_READS, PROBE_SWEEPS, PROBE_SEED)
        out[f"feas_{r:g}"] = res.n_feasible_reads / PROBE_READS
        rep = res.repaired_cost
        out[f"rep_{r:g}"] = 1.0 if rep != rep else rep / greedy - 1.0      # NaN (no facility) -> 1
    out["slope_lo"] = out["rep_0.05"] - out["rep_0.02"]
    out["slope_hi"] = out["rep_0.15"] - out["rep_0.05"]
    return out
