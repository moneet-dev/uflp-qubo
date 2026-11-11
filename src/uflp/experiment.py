"""One SA run on one instance at one penalty ratio -> one CSV row; plus a timestamped CSV logger.

Penalty: P = r * opt, as in the paper [P §4.5] (which used r in {0.25, 0.5, 0.75, 1.0}).
gap_pct uses the repaired cost; raw_gap_pct uses the best raw-feasible read, which is what
the paper measured (its solutions were never infeasible).
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .anneal import lp_warm_start, run_sa
from .io import Instance
from .qubo import build_penalty_function, build_q_matrix

COLUMNS = [
    "instance", "problem", "class", "m", "n", "opt", "r", "P", "seed", "reads", "sweeps",
    "builder", "warm_start", "best_energy", "raw_feasible", "n_feasible_reads",
    "n_no_facility_reads", "raw_best_cost", "raw_gap_pct", "repaired_cost", "gap_pct", "time_s",
    "cpu_time_s",
]


def fingerprint(inst: Instance) -> str:
    """Hash of (f, c): files that differ only in capacity/demand get the same value."""
    return hashlib.sha1(inst.f.tobytes() + inst.c.tobytes()).hexdigest()[:12]


def gap(cost: float, opt: float) -> float:
    """Optimality gap in percent, (cost - opt) / opt * 100; NaN if no cost."""
    return float("nan") if np.isnan(cost) else 100.0 * (cost - opt) / opt


def run_one(inst: Instance, cls: str, problem: str, opt: float, r: float, seed: int,
            reads: int, sweeps: int, builder: str = "penalty", warm_start: bool = False) -> dict:
    """Build the QUBO with P = r * opt, run seeded SA, and return one result row."""
    p = r * opt
    bqm = (build_penalty_function if builder == "penalty" else build_q_matrix)(inst, p)
    init = lp_warm_start(inst) if warm_start else None
    res = run_sa(inst, bqm, reads=reads, sweeps=sweeps, seed=seed, initial_state=init)
    return {
        "instance": inst.name, "problem": problem, "class": cls, "m": inst.m, "n": inst.n,
        "opt": opt, "r": r, "P": p, "seed": seed, "reads": reads, "sweeps": sweeps,
        "builder": builder, "warm_start": warm_start, "best_energy": res.best_energy,
        "raw_feasible": res.raw_feasible, "n_feasible_reads": res.n_feasible_reads,
        "n_no_facility_reads": res.n_no_facility_reads, "raw_best_cost": res.raw_best_cost,
        "raw_gap_pct": gap(res.raw_best_cost, opt), "repaired_cost": res.repaired_cost,
        "gap_pct": gap(res.repaired_cost, opt), "time_s": round(res.time_s, 4),
        "cpu_time_s": round(res.cpu_time_s, 4),
    }


@dataclass
class CsvLog:
    """Append-only CSV in results/ with a timestamped name, flushed after every row."""

    path: Path

    @classmethod
    def new(cls, results_dir: Path, prefix: str) -> "CsvLog":
        """Create results/<prefix>_<YYYYmmdd_HHMMSS>.csv with a header; never overwrites."""
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        if path.exists():
            raise FileExistsError(path)
        with open(path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=COLUMNS).writeheader()
        return cls(path)

    def write(self, row: dict) -> None:
        """Append one row."""
        with open(self.path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=COLUMNS).writerow(row)
