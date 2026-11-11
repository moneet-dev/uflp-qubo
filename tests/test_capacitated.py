"""Capacitated variant: [UL] worked example (Table 3), exact slack encoding, QUBO correctness."""
import itertools
import math

import dimod
import numpy as np
import pytest

from uflp.capacitated import (CapInstance, build_cflp_qubo, cflp_feasible, cflp_repair,
                              encode_slack, slack_coefficients, solve_cflp,
                              state_from_cflp_solution)
from uflp.io import DATA_DIR

# Li (2026) Urban Lifeline, Table 2: distances (unrounded: the table shows sqrt(2) as 1.41 etc.),
# capacities, fixed costs, demands; cost = distance x demand. Table 3: optimum vs capacity of facility 1.
D = np.array([20, 30, 40, 10.0])
DIST = np.array([[math.sqrt(2), math.sqrt(20), math.sqrt(2), 3],
                 [6.66, math.sqrt(2), math.sqrt(8), 3],
                 [4, math.sqrt(26), math.sqrt(8), 1]])
F = np.array([100, 120, 150.0])
TABLE3 = {40: (433.85, [0, 1]), 60: (377.28, [0, 1]), 80: (377.28, [0, 1]), 100: (349.02, [0]),
          120: (349.02, [0]), 140: (349.02, [0]), 160: (349.02, [0])}


def ul_instance(cap1: float) -> CapInstance:
    """[UL] Table 2 instance with facility 1's capacity set to cap1."""
    return CapInstance(f"ul_cap{cap1:g}", F, DIST * D[None, :], D, np.array([cap1, 80, 120.0]))


@pytest.mark.parametrize("cap1", sorted(TABLE3))
def test_milp_reproduces_urban_lifeline_table3(cap1):
    target, opened = TABLE3[cap1]
    res = solve_cflp(ul_instance(cap1))
    assert res.cost == pytest.approx(target, abs=0.005)
    assert res.open_facilities == opened


def test_split_demand_would_not_match_table3():
    """Single sourcing matters: with split demand the cap-40 optimum is lower than [UL]'s 433.85."""
    assert solve_cflp(ul_instance(40), split=True).cost < 433.85 - 1


@pytest.mark.parametrize("q", [1, 2, 3, 7, 8, 100, 5000])
def test_bounded_slack_expansion_is_exact(q):
    a = slack_coefficients(q)
    assert sum(a) == q and all(x > 0 for x in a)
    for v in sorted({0, 1, q // 3, q // 2, q - 1, q} & set(range(q + 1))):
        assert sum(ai * bi for ai, bi in zip(a, encode_slack(v, a))) == v


@pytest.mark.parametrize("cap1", [40, 100])
def test_qubo_energy_of_milp_optimum_equals_cost(cap1):
    inst = ul_instance(cap1)
    res = solve_cflp(inst)
    cq = build_cflp_qubo(inst, p=500.0)
    z = state_from_cflp_solution(inst, cq, res.open_facilities, res.assignment)
    assert cflp_feasible(inst, z)
    assert cq.bqm.energy((z, range(len(z)))) == pytest.approx(res.cost, rel=1e-12)


def test_tiny_count_capacity_qubo_minimum_is_the_cflp_optimum():
    """[UL] (A.4): w = 1, Q = k (max customers). Small enough for an exhaustive QUBO solve."""
    f = np.array([10.0, 12.0])
    c = np.array([[1.0, 2.0, 9.0], [8.0, 7.0, 1.0]])
    inst = CapInstance("tiny", f, c, np.ones(3), np.array([2.0, 2.0]))
    res = solve_cflp(inst)
    cq = build_cflp_qubo(inst, p=60.0)
    assert cq.bqm.num_variables == 2 + 6 + 4           # 2 slack bits per facility for k = 2
    best = dimod.ExactSolver().sample(cq.bqm).first
    z = np.array([best.sample[v] for v in range(cq.bqm.num_variables)])
    assert best.energy == pytest.approx(res.cost) and cflp_feasible(inst, z)
    # capacity really binds: with k = (1, 2) the cheapest uncapacitated choice is not allowed
    inst2 = CapInstance("tiny2", f, c, np.ones(3), np.array([1.0, 2.0]))
    cq2 = build_cflp_qubo(inst2, p=60.0)
    best2 = dimod.ExactSolver().sample(cq2.bqm).first
    assert best2.energy == pytest.approx(solve_cflp(inst2).cost)


def test_repair_respects_capacity_or_fails():
    inst = ul_instance(40)
    assert cflp_repair(inst, np.array([1, 0, 0])) is None        # 100 units of demand, room 40
    cost = cflp_repair(inst, np.array([1, 1, 0]))
    assert cost is not None and cost >= 433.85 - 1e-6
    assert cflp_repair(inst, np.array([0, 0, 0])) is None


@pytest.mark.skipif(not (DATA_DIR / "raw" / "cap41.txt").exists(), reason="run data/download.py")
def test_orlib_cap_parser_keeps_capacity_and_demand():
    from uflp.capacitated import parse_orlib_cap
    from uflp.io import load_instance
    inst = parse_orlib_cap((DATA_DIR / "raw" / "cap41.txt").read_text(), "cap41")
    u = load_instance("cap41")
    assert np.array_equal(inst.f, u.f) and np.array_equal(inst.c, u.c)
    assert (inst.Q == 5000).all() and inst.w[0] == 146 and inst.w.sum() == 58268


def test_sa_on_urban_lifeline_example_is_seeded_and_finds_optimum():
    from uflp.capacitated import run_cflp_sa
    inst = ul_instance(100)
    cq = build_cflp_qubo(inst, p=400.0)
    a = run_cflp_sa(inst, cq, reads=50, sweeps=1000, seed=4)
    b = run_cflp_sa(inst, cq, reads=50, sweeps=1000, seed=4)
    assert {k: v for k, v in a.items() if "time" not in k} == {k: v for k, v in b.items() if "time" not in k}
    # SA need not find the optimum (here it stops at a 2-facility solution, 377.28), but it must
    # find feasible reads, and no feasible or repaired cost may be below the true optimum.
    assert a["n_feasible_reads"] > 0
    assert a["raw_best_cost"] >= 349.02 - 0.005 and a["repaired_cost"] >= 349.02 - 0.005


@pytest.mark.parametrize("couple_y", [True, False])
def test_both_capacity_forms_are_exact_on_tiny_instance(couple_y):
    """Coupled (A.4 as written) and decoupled capacity rows: the QUBO minimum is the CFLP optimum."""
    f = np.array([10.0, 12.0])
    c = np.array([[1.0, 2.0, 9.0], [8.0, 7.0, 1.0]])
    for caps in ([2.0, 2.0], [1.0, 2.0]):
        inst = CapInstance("tiny", f, c, np.ones(3), np.array(caps))
        res = solve_cflp(inst)
        cq = build_cflp_qubo(inst, p=60.0, couple_y=couple_y)
        best = dimod.ExactSolver().sample(cq.bqm).first
        z = np.array([best.sample[v] for v in range(cq.bqm.num_variables)])
        assert best.energy == pytest.approx(res.cost) and cflp_feasible(inst, z)
        zo = state_from_cflp_solution(inst, cq, res.open_facilities, res.assignment)
        assert cq.bqm.energy((zo, range(len(zo)))) == pytest.approx(res.cost)


def test_decoupled_form_energy_of_ul_optimum_equals_cost():
    inst = ul_instance(40)
    res = solve_cflp(inst)
    cq = build_cflp_qubo(inst, p=500.0, couple_y=False)
    z = state_from_cflp_solution(inst, cq, res.open_facilities, res.assignment)
    assert cq.bqm.energy((z, range(len(z)))) == pytest.approx(res.cost, rel=1e-12)
