"""Exact UFLP solve as a mixed-integer linear program (MILP) with PuLP + HiGHS.

A MILP is a linear objective and linear constraints where some variables must be
integers; HiGHS solves it exactly by branch and bound (split on a fractional variable,
solve the two linear sub-problems, prune branches whose bound is already worse than
the best solution found). The LP relaxation drops integrality (0 <= x, y <= 1); its
value is a lower bound on the true optimum and is cheap to compute.

Formulation [P eqs. 10-13]:

    minimize    sum_i f_i y_i + sum_ij c_ij x_ij
    subject to  sum_i x_ij = 1      for every customer j
                x_ij <= y_i         for every facility i, customer j
                x_ij, y_i in {0, 1}
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pulp

from .io import Instance


@dataclass(frozen=True)
class MilpResult:
    """Outcome of an exact (or relaxed) solve."""

    cost: float
    open_facilities: list[int]   # indices i with y_i = 1 (empty for the LP relaxation)
    assignment: list[int]        # assignment[j] = facility serving customer j (empty for LP)
    status: str
    time_s: float


def build_model(f: np.ndarray, c: np.ndarray, relax: bool = False):
    """Build the PuLP model; relax=True gives the LP relaxation (continuous 0..1 variables)."""
    m, n = c.shape
    cat = "Continuous" if relax else "Binary"
    prob = pulp.LpProblem("UFLP", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y{i}", 0, 1, cat=cat) for i in range(m)]
    x = [[pulp.LpVariable(f"x{i}_{j}", 0, 1, cat=cat) for j in range(n)] for i in range(m)]
    prob += pulp.lpSum(float(f[i]) * y[i] for i in range(m)) + pulp.lpSum(
        float(c[i, j]) * x[i][j] for i in range(m) for j in range(n))
    for j in range(n):
        prob += pulp.lpSum(x[i][j] for i in range(m)) == 1, f"assign_{j}"
    for i in range(m):
        for j in range(n):
            prob += x[i][j] <= y[i], f"link_{i}_{j}"
    return prob, y, x


def solve_uflp(inst: Instance, relax: bool = False, time_limit: float | None = None) -> MilpResult:
    """Solve the UFLP exactly (or its LP relaxation) with HiGHS and return cost and solution."""
    prob, y, x = build_model(inst.f, inst.c, relax=relax)
    # No threads option (see capacitated.solve_cflp): mixing thread counts in one process fails.
    solver = pulp.HiGHS(msg=False, timeLimit=time_limit, mip_rel_gap=0.0) \
        if not relax else pulp.HiGHS(msg=False, timeLimit=time_limit)
    t0 = time.perf_counter()
    prob.solve(solver)
    elapsed = time.perf_counter() - t0
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"{inst.name}: HiGHS status {status}")
    if relax:
        return MilpResult(pulp.value(prob.objective), [], [], status, elapsed)
    opened = [i for i in range(inst.m) if y[i].value() > 0.5]
    assign = [next(i for i in range(inst.m) if x[i][j].value() > 0.5) for j in range(inst.n)]
    # Recompute the cost from the 0/1 solution so solver round-off cannot leak in.
    cost = solution_cost(inst, opened, assign)
    return MilpResult(cost, opened, assign, status, elapsed)


def lp_bound(inst: Instance) -> float:
    """LP-relaxation value: a lower bound on the UFLP optimum, used later as a feature."""
    return solve_uflp(inst, relax=True).cost


def solution_cost(inst: Instance, opened: list[int], assignment: list[int]) -> float:
    """Cost of a solution: fixed costs of opened sites plus each customer's service cost."""
    if any(a not in set(opened) for a in assignment):
        raise ValueError("a customer is assigned to a closed facility")
    return float(inst.f[opened].sum() + inst.c[assignment, np.arange(inst.n)].sum())
