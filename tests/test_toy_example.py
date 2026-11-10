"""Acceptance test: the worked example in the appendix of
Li, Hennig, Kochenberger & Glover (2026), Annals of Operations Research,
"Quantum bridge analytics: a QUBO approach to the uncapacitated facility location problem".

Paper's stated results for this example:
  * optimal cost 290, with only facility 3 open and every customer assigned to it
  * with P = 250 the QUBO minimum is -710; adding the constant P*|J| = 1000 gives 290

Tested on 9 Nov 2026 with pulp 3.3.2 + highspy, dimod 0.12.23, dwave-samplers 1.8.0.
Run with:  pytest -q tests/test_toy_example.py
"""
import itertools

import dimod
import pulp
from dwave.samplers import SimulatedAnnealingSampler

F = [100, 125, 125]            # fixed cost of facilities 1..3 ($ thousand)
C = [[35, 40, 60, 60],         # C[i][j]: cost for facility i to serve customer j
     [40, 55, 50, 60],
     [40, 30, 45, 50]]
P = 250
M, N = len(F), len(C[0])


def solve_milp(f, c):
    """Exact UFLP solve (paper eqs. 10-13) with PuLP + HiGHS."""
    m, n = len(f), len(c[0])
    prob = pulp.LpProblem("UFLP", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y{i}", cat="Binary") for i in range(m)]
    x = [[pulp.LpVariable(f"x{i}_{j}", cat="Binary") for j in range(n)] for i in range(m)]
    prob += pulp.lpSum(f[i] * y[i] for i in range(m)) + pulp.lpSum(
        c[i][j] * x[i][j] for i in range(m) for j in range(n))
    for j in range(n):
        prob += pulp.lpSum(x[i][j] for i in range(m)) == 1
    for i in range(m):
        for j in range(n):
            prob += x[i][j] <= y[i]
    prob.solve(pulp.HiGHS(msg=False))
    assert pulp.LpStatus[prob.status] == "Optimal"
    sol = {v.name: int(round(v.value())) for v in prob.variables()}
    return pulp.value(prob.objective), sol


def qubo_penalty_function(f, c, p):
    """QUBO of paper eq. 15 built algebraically; dimod keeps the constant p*|J| as bqm.offset."""
    m, n = len(f), len(c[0])
    bqm = dimod.BinaryQuadraticModel("BINARY")
    for i in range(m):
        bqm.add_linear(f"y{i}", f[i])
        for j in range(n):
            bqm.add_linear(f"x{i}_{j}", c[i][j])
    for j in range(n):                      # p * (1 - sum_i x_ij)^2
        bqm.add_linear_equality_constraint(
            [(f"x{i}_{j}", 1) for i in range(m)], constant=-1, lagrange_multiplier=p)
    for i in range(m):                      # p * (x_ij - x_ij * y_i), known penalty for x <= y
        for j in range(n):
            bqm.add_linear(f"x{i}_{j}", p)
            bqm.add_quadratic(f"x{i}_{j}", f"y{i}", -p)
    return bqm


def qubo_q_matrix(f, c, p):
    """QUBO built entry by entry from the paper's Properties 1 and 2 (no constant term)."""
    m, n = len(f), len(c[0])
    q = {}
    for i in range(m):
        q[(f"y{i}", f"y{i}")] = f[i]
        for j in range(n):
            q[(f"x{i}_{j}", f"x{i}_{j}")] = c[i][j]
            q[(f"y{i}", f"x{i}_{j}")] = -p
    for j in range(n):
        for i, k in itertools.combinations(range(m), 2):
            q[(f"x{i}_{j}", f"x{k}_{j}")] = 2 * p
    return dimod.BinaryQuadraticModel.from_qubo(q)


def test_milp_matches_paper():
    cost, sol = solve_milp(F, C)
    assert cost == 290
    assert [i + 1 for i in range(M) if sol[f"y{i}"] == 1] == [3]


def test_two_qubo_constructions_agree():
    a, b = qubo_penalty_function(F, C, P), qubo_q_matrix(F, C, P)
    assert a.linear == b.linear and a.quadratic == b.quadratic
    assert a.offset == P * N == 1000


def test_exact_qubo_minimum_matches_paper():
    best = dimod.ExactSolver().sample(qubo_q_matrix(F, C, P)).first   # 2^15 states
    assert best.energy == -710
    assert best.energy + P * N == 290


def test_milp_optimum_scores_its_cost_inside_qubo():
    cost, sol = solve_milp(F, C)
    assert qubo_penalty_function(F, C, P).energy(sol) == cost


def test_simulated_annealing_finds_optimum_on_toy():
    ss = SimulatedAnnealingSampler().sample(
        qubo_penalty_function(F, C, P), num_reads=100, num_sweeps=1000, seed=42)
    assert ss.first.energy == 290
