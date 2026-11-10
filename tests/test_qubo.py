"""QUBO tests: both builders agree; energies equal real costs; infeasible states pay >= P; SA/repair basics."""
import numpy as np
import pytest

from uflp.anneal import lp_warm_start, run_sa
from uflp.io import DATA_DIR, Instance, load_instance, load_optima
from uflp.milp import solve_uflp
from uflp.qubo import (build_penalty_function, build_q_matrix, energies, is_feasible,
                       state_from_solution)
from uflp.repair import repair_cost, repair_costs

TOY = Instance("toy", f=np.array([100.0, 125, 125]),
               c=np.array([[35.0, 40, 60, 60], [40, 55, 50, 60], [40, 30, 45, 50]]))
HAVE_DATA = (DATA_DIR / "optima.csv").exists()
OPTIMA = load_optima() if HAVE_DATA else []


def assert_same_model(a, b, rel=1e-12):
    """Same variables, linear and quadratic biases, and offset (up to float round-off)."""
    assert set(a.variables) == set(b.variables)
    assert a.num_interactions == b.num_interactions
    for v, bias in a.linear.items():
        assert b.linear[v] == pytest.approx(bias, rel=rel, abs=1e-9)
    for (u, v), bias in a.quadratic.items():
        assert b.quadratic[(u, v)] == pytest.approx(bias, rel=rel)
    assert a.offset == pytest.approx(b.offset, rel=rel)


def test_toy_builders_agree_and_match_paper_matrix():
    a, b = build_penalty_function(TOY, 250), build_q_matrix(TOY, 250)
    assert_same_model(a, b)
    assert a.offset == 1000
    # [P appendix] row of x_11 (label 3): 2P with x_21 (label 7) and x_31 (label 11), -P with y_1
    assert b.quadratic[(3, 7)] == 500 and b.quadratic[(3, 11)] == 500 and b.quadratic[(0, 3)] == -250
    assert (3, 14) not in b.quadratic   # the clamped entry the printed Property 2 formula would add


def test_toy_energy_of_optimum_is_290():
    z = state_from_solution(TOY, [2], [2, 2, 2, 2])
    assert is_feasible(TOY, z)
    assert energies(build_q_matrix(TOY, 250), z)[0] == 290


def test_infeasible_states_pay_at_least_p():
    p = 7.0
    bqm = build_q_matrix(TOY, p)
    z = state_from_solution(TOY, [0], [0, 0, 0, 0])
    z[0] = 0                                   # close facility 1 but keep customers on it
    assert not is_feasible(TOY, z)
    base = float(TOY.c[0].sum())               # objective part of that state
    assert energies(bqm, z)[0] >= base + p


def test_repair():
    assert repair_cost(TOY, np.array([0, 0, 1])) == 290
    assert repair_cost(TOY, np.array([0, 0, 0])) is None
    out = repair_costs(TOY, np.array([[0, 0, 1], [0, 0, 0], [1, 0, 1]]))
    assert out[0] == 290 and np.isnan(out[1]) and out[2] == 100 + 125 + 35 + 30 + 45 + 50


def test_sa_on_toy_is_seeded_and_finds_optimum():
    bqm = build_penalty_function(TOY, 250)
    r1 = run_sa(TOY, bqm, reads=50, sweeps=500, seed=3)
    r2 = run_sa(TOY, bqm, reads=50, sweeps=500, seed=3)
    assert r1.best_energy == r2.best_energy == 290
    assert r1.n_feasible_reads == r2.n_feasible_reads
    assert r1.raw_feasible and r1.raw_best_cost == 290 and r1.repaired_cost == 290


def test_lp_warm_start_is_feasible_and_optimal_on_toy():
    z = lp_warm_start(TOY)
    assert is_feasible(TOY, z)
    assert energies(build_q_matrix(TOY, 250), z)[0] == 290


@pytest.mark.data
@pytest.mark.skipif(not HAVE_DATA, reason="run data/download.py first")
@pytest.mark.parametrize("row", OPTIMA, ids=[r["name"] for r in OPTIMA])
def test_builders_agree_and_energy_of_optimum_equals_cost(row):
    """M2 acceptance on every file, at the paper's 50% penalty."""
    inst = load_instance(row["name"])
    p = 0.5 * row["opt"]
    a, b = build_penalty_function(inst, p), build_q_matrix(inst, p)
    assert_same_model(a, b)
    assert a.num_variables == inst.m * inst.n + inst.m
    res = solve_uflp(inst)
    z = state_from_solution(inst, res.open_facilities, res.assignment)
    assert energies(a, z)[0] == pytest.approx(row["opt"], rel=1e-9)
    assert energies(b, z)[0] == pytest.approx(row["opt"], rel=1e-9)
