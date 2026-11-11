"""Capacitated facility location (CFLP): exact MILP and a QUBO with slack variables.

The model follows Li (2026), Urban Lifeline 4:7, Appendix A [UL], with single sourcing
(each customer is served by exactly one facility), written with a general capacity weight:

    minimize    sum_i f_i y_i + sum_ij c_ij x_ij
    subject to  sum_i x_ij = 1                    for every customer j
                x_ij <= y_i                       for every i, j
                sum_j w_j x_ij <= Q_i y_i         for every facility i   (capacity)
                x_ij, y_i in {0, 1}

With w_j = 1 and Q_i = k_i this is [UL] (A.4), a limit on the *number* of customers. With
w_j = demand_j it is the demand-based capacity of [UL]'s worked example (Table 2) and of the
ORLIB cap files. Both are covered by the same code.

QUBO. An inequality cannot be squared directly, so it is first made an equality with a slack
variable s_i >= 0:  sum_j w_j x_ij + s_i = Q_i y_i  [P eq. 8].  The slack is an integer, so it
is written in binary, s_i = sum_b a_b z_ib with weights 1, 2, 4, ... and a last weight chosen so
the largest value is exactly Q_i ("bounded binary expansion"), and the equality is then penalized
as P_cap * (sum_j w_j x_ij + s_i - Q_i y_i)^2. That adds about log2(Q_i) binary variables per
facility, which is why [P §5] lists general inequality constraints as a scaling challenge.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import dimod
import numpy as np
import pulp

from .io import Instance
from .qubo import build_penalty_function


@dataclass(frozen=True)
class CapInstance:
    """A CFLP instance: fixed costs f (m,), service costs c (m, n), weights w (n,), capacities Q (m,)."""

    name: str
    f: np.ndarray
    c: np.ndarray
    w: np.ndarray
    Q: np.ndarray

    @property
    def m(self) -> int:
        """Number of candidate facility sites."""
        return int(self.f.shape[0])

    @property
    def n(self) -> int:
        """Number of customers."""
        return int(self.c.shape[1])


def parse_orlib_cap(text: str, name: str = "") -> CapInstance:
    """ORLIB cap file with capacities (demand units) and demands kept; c transposed to (m, n)."""
    toks = text.split()
    m, n = int(toks[0]), int(toks[1])
    pos = 2
    Q, f = np.empty(m), np.empty(m)
    for i in range(m):
        Q[i], f[i] = float(toks[pos]), float(toks[pos + 1])
        pos += 2
    w, c_t = np.empty(n), np.empty((n, m))
    for j in range(n):
        w[j] = float(toks[pos])
        c_t[j] = [float(t) for t in toks[pos + 1: pos + 1 + m]]
        pos += 1 + m
    if pos != len(toks):
        raise ValueError(f"{name}: {len(toks) - pos} unread tokens")
    return CapInstance(name, f, c_t.T.copy(), w, Q)


@dataclass(frozen=True)
class CflpResult:
    """Exact (single-source) or split-demand solution of a CFLP."""

    cost: float
    open_facilities: list[int]
    assignment: list[int]        # facility per customer (empty when demand may be split)
    status: str
    time_s: float


def solve_cflp(inst: CapInstance, split: bool = False, time_limit: float | None = None) -> CflpResult:
    """Exact MILP (HiGHS). split=True lets a customer's demand be divided among facilities."""
    m, n = inst.m, inst.n
    prob = pulp.LpProblem("CFLP", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y{i}", cat="Binary") for i in range(m)]
    xcat = "Continuous" if split else "Binary"
    x = [[pulp.LpVariable(f"x{i}_{j}", 0, 1, cat=xcat) for j in range(n)] for i in range(m)]
    prob += pulp.lpSum(float(inst.f[i]) * y[i] for i in range(m)) + pulp.lpSum(
        float(inst.c[i, j]) * x[i][j] for i in range(m) for j in range(n))
    for j in range(n):
        prob += pulp.lpSum(x[i][j] for i in range(m)) == 1
    for i in range(m):
        for j in range(n):
            prob += x[i][j] <= y[i]
        prob += pulp.lpSum(float(inst.w[j]) * x[i][j] for j in range(n)) <= float(inst.Q[i]) * y[i]
    t0 = time.perf_counter()
    prob.solve(pulp.HiGHS(msg=False, timeLimit=time_limit, mip_rel_gap=0.0, threads=1))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"{inst.name}: HiGHS status {status}")
    opened = [i for i in range(m) if y[i].value() > 0.5]
    if split:
        return CflpResult(float(pulp.value(prob.objective)), opened, [], status, time.perf_counter() - t0)
    assign = [next(i for i in range(m) if x[i][j].value() > 0.5) for j in range(n)]
    return CflpResult(cflp_cost(inst, opened, assign), opened, assign, status, time.perf_counter() - t0)


def slack_coefficients(q: int) -> list[int]:
    """Bounded binary expansion of an integer slack in [0, q]: 1, 2, 4, ..., then q - (2^K - 1)."""
    if q <= 0:
        return []
    k = int(math.floor(math.log2(q)))
    coeffs = [2 ** b for b in range(k)]
    coeffs.append(q - (2 ** k - 1))
    return coeffs


def encode_slack(value: int, coeffs: list[int]) -> list[int]:
    """Bits z with sum(a_b z_b) == value for the bounded expansion (value in [0, sum(coeffs)])."""
    if not 0 <= value <= sum(coeffs):
        raise ValueError("slack out of range")
    bits = [0] * len(coeffs)
    if coeffs and value >= coeffs[-1] and value - coeffs[-1] <= sum(coeffs[:-1]):
        bits[-1], value = 1, value - coeffs[-1]
    for b in range(len(coeffs) - 2, -1, -1):
        if value >= coeffs[b]:
            bits[b], value = 1, value - coeffs[b]
    if value != 0:
        raise ValueError("value not representable")
    return bits


@dataclass(frozen=True)
class CapQubo:
    """The CFLP QUBO plus where its slack bits sit (labels after the m + m*n UFLP variables)."""

    bqm: dimod.BinaryQuadraticModel
    slack_labels: list[list[int]]       # per facility, labels of its slack bits
    slack_coeffs: list[list[int]]       # per facility, the matching weights
    scale: float                        # capacity rows are divided by this (mean weight)
    couple_y: bool = True               # capacity row uses Q_i * y_i (True) or Q_i (False)


def build_cflp_qubo(inst: CapInstance, p: float, couple_y: bool = True) -> CapQubo:
    """Eq. 15 of [P] plus a squared capacity row per facility, scaled by the mean weight.

    couple_y=True  ("coupled", [UL] (A.4) as written):  P * ((sum_j w_j x_ij + s_i - Q_i y_i)/scale)^2
    couple_y=False ("decoupled"):                        P * ((sum_j w_j x_ij + s_i - Q_i)/scale)^2
    Both are exact for single sourcing: in the decoupled form a closed facility serves nobody because
    of the x <= y penalty already in eq. 15, and its slack simply equals Q_i. The difference matters
    for single-flip search: in the coupled form, closing a facility whose load plus slack fills its
    capacity changes the row by Q_i and costs P * (Q_i/scale)^2 in one flip, so facilities stay open.

    Labels: y_i -> i, x_ij -> m + i*n + j (as in qubo.py), then the slack bits of each facility.
    Weights and capacities must be integers so the slack can be exact.
    """
    if not (np.allclose(inst.w, np.round(inst.w)) and np.allclose(inst.Q, np.round(inst.Q))):
        raise ValueError("weights and capacities must be integers for an exact slack")
    m, n = inst.m, inst.n
    bqm = build_penalty_function(Instance(inst.name, inst.f, inst.c), p)
    scale = float(inst.w.mean())
    nxt = m + m * n
    labels, coeffs = [], []
    for i in range(m):
        a = slack_coefficients(int(round(inst.Q[i])))
        lab = list(range(nxt, nxt + len(a)))
        nxt += len(a)
        labels.append(lab)
        coeffs.append(a)
        terms = [(m + i * n + j, float(inst.w[j]) / scale) for j in range(n)]
        terms += [(v, ab / scale) for v, ab in zip(lab, a)]
        if couple_y:
            terms.append((i, -float(inst.Q[i]) / scale))
            const = 0.0
        else:
            const = -float(inst.Q[i]) / scale
        bqm.add_linear_equality_constraint(terms, lagrange_multiplier=p, constant=const)
    return CapQubo(bqm, labels, coeffs, scale, couple_y)


def state_from_cflp_solution(inst: CapInstance, cq: CapQubo, opened: list[int],
                             assignment: list[int]) -> np.ndarray:
    """0/1 state for a feasible single-source solution, with each slack set to its exact value."""
    z = np.zeros(cq.bqm.num_variables, dtype=np.int8)
    z[opened] = 1
    for j, i in enumerate(assignment):
        z[inst.m + i * inst.n + j] = 1
    load = np.zeros(inst.m)
    np.add.at(load, assignment, inst.w)
    for i in range(inst.m):
        s = int(round(inst.Q[i] - load[i])) if (i in set(opened) or not cq.couple_y) else 0
        z[cq.slack_labels[i]] = encode_slack(s, cq.slack_coeffs[i])
    return z


def cflp_feasible(inst: CapInstance, z: np.ndarray) -> bool:
    """Original constraints only: one facility per customer, open facilities, capacities respected."""
    y = z[: inst.m]
    x = z[inst.m: inst.m + inst.m * inst.n].reshape(inst.m, inst.n)
    if not (np.all(x.sum(axis=0) == 1) and np.all(x <= y[:, None])):
        return False
    return bool(np.all(x @ inst.w <= inst.Q * y + 1e-9))


def cflp_repair(inst: CapInstance, y: np.ndarray) -> float | None:
    """Keep the opened facilities; assign customers largest demand first to the cheapest open
    facility with enough room left. Returns the cost, or None if some customer does not fit."""
    opened = np.flatnonzero(np.asarray(y) > 0)
    if opened.size == 0:
        return None
    room = inst.Q[opened].astype(float).copy()
    cost = float(inst.f[opened].sum())
    for j in np.argsort(-inst.w, kind="stable"):
        order = np.argsort(inst.c[opened, j], kind="stable")
        for k in order:
            if room[k] >= inst.w[j] - 1e-9:
                room[k] -= inst.w[j]
                cost += float(inst.c[opened[k], j])
                break
        else:
            return None
    return cost


def run_cflp_sa(inst: CapInstance, cq: CapQubo, reads: int, sweeps: int, seed: int) -> dict:
    """Seeded SA on the CFLP QUBO; summarize energy, raw feasibility, raw and repaired cost.

    A read can satisfy every original constraint while its slack bits do not match the spare
    capacity; its energy then exceeds its cost. So raw cost is computed from (x, y) directly, and
    reads whose energy equals their cost (penalty zero) are counted separately as slack-exact.
    """
    from dwave.samplers import SimulatedAnnealingSampler
    nv = cq.bqm.num_variables
    labels = list(range(nv))
    t0, c0 = time.perf_counter(), time.process_time()
    ss = SimulatedAnnealingSampler().sample(cq.bqm, num_reads=reads, num_sweeps=sweeps, seed=seed)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    states = dimod.as_samples(ss)[0][:, [ss.variables.index(v) for v in labels]]
    energy = cq.bqm.energies((states, labels))
    m, n = inst.m, inst.n
    feas, raw_costs, repaired = [], [], []
    for z in states:
        ok = cflp_feasible(inst, z)
        feas.append(ok)
        if ok:
            x = z[m: m + m * n].reshape(m, n)
            raw_costs.append(float(inst.f @ z[:m] + (inst.c * x).sum()))
        repaired.append(cflp_repair(inst, z[:m]))
    feas = np.array(feas)
    exact = feas & np.isclose(energy, [float(inst.f @ z[:m] + (inst.c * z[m: m + m * n].reshape(m, n)).sum())
                                       for z in states], rtol=1e-12, atol=1e-6)
    rep = [r for r in repaired if r is not None]
    return {"n_vars": nv, "best_energy": float(energy.min()), "n_feasible_reads": int(feas.sum()),
            "n_slack_exact_reads": int(exact.sum()),
            "raw_best_cost": min(raw_costs) if raw_costs else float("nan"),
            "repaired_cost": min(rep) if rep else float("nan"),
            "n_repair_fail": int(sum(r is None for r in repaired)),
            "time_s": round(wall, 4), "cpu_time_s": round(cpu, 4)}


def cflp_cost(inst: CapInstance, opened: list[int], assignment: list[int]) -> float:
    """Cost of a single-source solution; raises if it breaks a capacity or uses a closed site."""
    load = np.zeros(inst.m)
    np.add.at(load, assignment, inst.w)
    if any(a not in set(opened) for a in assignment):
        raise ValueError("customer assigned to a closed facility")
    if np.any(load > inst.Q + 1e-9):
        raise ValueError("capacity exceeded")
    return float(inst.f[opened].sum() + inst.c[assignment, np.arange(inst.n)].sum())
