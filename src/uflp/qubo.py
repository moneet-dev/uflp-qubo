"""QUBO models of the UFLP: two independent constructions that must give the same model.

A QUBO (quadratic unconstrained binary optimization) problem minimizes z^T Q z over 0/1
vectors z, with no constraints. The UFLP constraints are moved into the objective as
penalties: P * (violation), which is zero for feasible solutions and at least P otherwise.

Variables use the paper's order [P §4.2]: z = [y_1..y_m, x_11..x_1n, ..., x_m1..x_mn], so
y_i has label i and x_ij has label m + i*n + j (0-based).

Both builders keep the constant P*|J| as ``bqm.offset``, so the energy of a feasible
solution equals its real UFLP cost (paper appendix: -710 + 1,000 = 290).
"""
from __future__ import annotations

import dimod
import numpy as np

from .io import Instance


def x_label(m: int, n: int, i: int, j: int) -> int:
    """Integer label of x_ij in the paper's variable order."""
    return m + i * n + j


def build_penalty_function(inst: Instance, p: float) -> dimod.BinaryQuadraticModel:
    """Paper eq. 15 built algebraically: objective + P*sum_j (1 - sum_i x_ij)^2 + P*sum_ij (x_ij - x_ij y_i)."""
    m, n = inst.m, inst.n
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    bqm.add_linear_from((i, float(inst.f[i])) for i in range(m))
    bqm.add_linear_from((x_label(m, n, i, j), float(inst.c[i, j]))
                        for i in range(m) for j in range(n))
    for j in range(n):  # each customer assigned exactly once: P * (sum_i x_ij - 1)^2
        bqm.add_linear_equality_constraint(
            [(x_label(m, n, i, j), 1.0) for i in range(m)], lagrange_multiplier=p, constant=-1.0)
    for i in range(m):  # x_ij <= y_i via the known exact penalty P * (x - x*y) [T]
        for j in range(n):
            bqm.add_linear(x_label(m, n, i, j), p)
            bqm.add_quadratic(x_label(m, n, i, j), i, -p)
    return bqm


def build_q_matrix(inst: Instance, p: float) -> dimod.BinaryQuadraticModel:
    """Q matrix entry by entry from [P] Properties 1-2 (meaning, not the printed index formulas).

    Diagonal: f_i for y_i, c_ij for x_ij. Off-diagonal: -P for (y_i, x_ij); 2P for
    (x_ij, x_i'j) with i < i'. The paper's Q has no constant; we attach P*|J| as the offset.
    """
    m, n = inst.m, inst.n
    linear = np.concatenate([inst.f, inst.c.ravel()]).astype(float)
    ii, jj = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    xl = m + ii * n + jj                                   # labels of x_ij, shape (m, n)
    rows = [ii.ravel()]                                    # (y_i, x_ij): -P
    cols = [xl.ravel()]
    vals = [np.full(m * n, -p)]
    a, b = np.triu_indices(m, k=1)                         # facility pairs i < i'
    rows.append(xl[a].ravel())                             # (x_ij, x_i'j): 2P, same customer j
    cols.append(xl[b].ravel())
    vals.append(np.full(a.size * n, 2 * p))
    quad = (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals))
    return dimod.BinaryQuadraticModel.from_numpy_vectors(linear, quad, p * n, dimod.BINARY)


def penalty_bounds(inst: Instance, opt: float | None = None) -> tuple[float, float]:
    """Bounds on P for the QUBO minimum to be the UFLP optimum (an "exact" penalty).

    Necessary: P >= max_j min_i c_ij (else dropping that customer is cheaper than serving it)
    and, if opt is known, P >= opt/|J| (else dropping every customer is cheaper).
    Sufficient: P > max(max_j min_i (f_i + c_ij), max_i f_i): then every infeasible state has
    an improving move (serve a dropped customer, drop a duplicate assignment, or open a
    facility that is serving customers), so every minimizer is feasible.
    Returns (necessary, sufficient). The sufficient bound needs no optimum.
    """
    nec = float(inst.c.min(axis=0).max())
    if opt is not None:
        nec = max(nec, opt / inst.n)
    suf = float(max((inst.f[:, None] + inst.c).min(axis=0).max(), inst.f.max()))
    return nec, suf


def state_from_solution(inst: Instance, opened: list[int], assignment: list[int]) -> np.ndarray:
    """0/1 vector z for a UFLP solution (opened facilities, facility of each customer)."""
    z = np.zeros(inst.m + inst.m * inst.n, dtype=np.int8)
    z[opened] = 1
    for j, i in enumerate(assignment):
        z[x_label(inst.m, inst.n, i, j)] = 1
    return z


def decode(inst: Instance, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 0/1 state (in label order 0..m+mn-1) into y (m,) and x (m, n)."""
    z = np.asarray(z)
    return z[: inst.m], z[inst.m:].reshape(inst.m, inst.n)


def is_feasible(inst: Instance, z: np.ndarray) -> bool:
    """True if every customer has exactly one facility and only open facilities serve."""
    y, x = decode(inst, z)
    return bool(np.all(x.sum(axis=0) == 1) and np.all(x <= y[:, None]))


def energies(bqm: dimod.BinaryQuadraticModel, states: np.ndarray) -> np.ndarray:
    """QUBO energy (offset included) of each row of `states`, columns in label order."""
    return bqm.energies((np.atleast_2d(states), list(range(states.shape[-1]))))
