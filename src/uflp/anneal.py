"""Simulated annealing (SA) on a UFLP QUBO, plus an optional LP warm start.

SA is a classical local search: it flips one variable at a time, always accepting moves
that lower the energy and accepting uphill moves with probability exp(-beta * increase),
where beta (inverse temperature) grows during the run so the search settles into a low
valley. One "sweep" proposes a flip of every variable once. These are classical results,
never quantum results.

Warm start ([P §3.2], "Gap 2"): start SA from the LP-relaxation solution made feasible.
On ORLIB the LP relaxation is already integral and optimal (CLAUDE.md §7.1), so a warm
start there begins at the optimum; use it only for deliberate comparisons.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import dimod
import numpy as np
import pulp
from dwave.samplers import SimulatedAnnealingSampler

from .io import Instance
from .milp import build_model
from .qubo import is_feasible, state_from_solution
from .repair import repair_costs


@dataclass(frozen=True)
class SAResult:
    """Summary of one SA call (many reads) on one QUBO."""

    best_energy: float            # lowest energy over reads, offset included
    raw_feasible: bool            # is the lowest-energy read itself a feasible UFLP solution?
    n_feasible_reads: int
    raw_best_cost: float          # best energy among feasible reads (= their real cost); NaN if none
    repaired_cost: float          # best repaired cost over reads; NaN if no read opened a facility
    n_no_facility_reads: int
    time_s: float


def run_sa(inst: Instance, bqm: dimod.BinaryQuadraticModel, reads: int, sweeps: int, seed: int,
           initial_state: np.ndarray | None = None) -> SAResult:
    """Run seeded SA and summarize raw feasibility, raw cost and repaired cost."""
    kwargs = {}
    if initial_state is not None:
        labels = list(range(len(initial_state)))
        kwargs = {"initial_states": (np.tile(initial_state, (reads, 1)), labels),
                  "initial_states_generator": "none"}
    t0 = time.perf_counter()
    ss = SimulatedAnnealingSampler().sample(
        bqm, num_reads=reads, num_sweeps=sweeps, seed=seed, **kwargs)
    elapsed = time.perf_counter() - t0
    labels = list(range(inst.m + inst.m * inst.n))
    states = dimod.as_samples(ss)[0]
    order = [ss.variables.index(v) for v in labels]
    states = states[:, order]
    en = ss.record.energy
    feas = np.array([is_feasible(inst, z) for z in states])
    rep = repair_costs(inst, states[:, : inst.m])
    best = int(np.argmin(en))
    return SAResult(
        best_energy=float(en[best]),
        raw_feasible=bool(feas[best]),
        n_feasible_reads=int(feas.sum()),
        raw_best_cost=float(en[feas].min()) if feas.any() else float("nan"),
        repaired_cost=float(np.nanmin(rep)) if np.isfinite(rep).any() else float("nan"),
        n_no_facility_reads=int(np.isnan(rep).sum()),
        time_s=elapsed,
    )


def lp_warm_start(inst: Instance) -> np.ndarray:
    """LP-relaxation solution made feasible: open sites with y_i >= 0.5 (else the largest), then repair."""
    prob, y, _ = build_model(inst.f, inst.c, relax=True)
    prob.solve(pulp.HiGHS(msg=False))
    yv = np.array([v.value() for v in y])
    opened = np.flatnonzero(yv >= 0.5)
    if opened.size == 0:
        opened = np.array([int(np.argmax(yv))])
    assign = opened[inst.c[opened].argmin(axis=0)]
    return state_from_solution(inst, opened.tolist(), assign.tolist())
