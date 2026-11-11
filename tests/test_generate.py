"""Generator, greedy heuristic and feature tests."""
import math

import numpy as np
import pytest

from uflp.features import features
from uflp.generate import FAMILIES, design, generate
from uflp.heuristics import greedy_add
from uflp.io import Instance
from uflp.milp import lp_bound, solve_uflp

TOY = Instance("toy", f=np.array([100.0, 125, 125]),
               c=np.array([[35.0, 40, 60, 60], [40, 55, 50, 60], [40, 30, 45, 50]]))


@pytest.mark.parametrize("family", FAMILIES)
def test_generator_is_deterministic_and_well_formed(family):
    a, b = generate(family, 20, 50, 2.0, seed=5), generate(family, 20, 50, 2.0, seed=5)
    assert np.array_equal(a.f, b.f) and np.array_equal(a.c, b.c)
    assert a.f.shape == (20,) and a.c.shape == (20, 50)
    assert (a.f >= 0).all() and (a.c >= 0).all()
    assert not np.array_equal(a.c, generate(family, 20, 50, 2.0, seed=6).c)


def test_gap_family_gives_each_customer_k_cheap_sites_and_a_positive_lp_gap():
    inst = generate("gap", 16, 50, 16.0, seed=3)
    assert ((inst.c <= 400).sum(axis=0) == 3).all()
    opt = solve_uflp(inst).cost
    assert (opt - lp_bound(inst)) / opt > 0.05


def test_design_is_seeded_and_balanced():
    d = design()
    assert d == design() and len(d) == 200
    assert {s["family"] for s in d} == set(FAMILIES)
    assert all(s["phi"] >= 8 for s in d if s["family"] == "gap")
    assert len({s["seed"] for s in d}) == 200


def test_greedy_on_toy_and_never_below_optimum():
    cost, opened = greedy_add(TOY)
    assert cost == 290 and opened == [2]
    for family in FAMILIES:
        inst = generate(family, 16, 50, 4.0, seed=1)
        g, _ = greedy_add(inst)
        assert g >= solve_uflp(inst).cost - 1e-6


def test_features_complete_and_free_of_the_optimum():
    ft = features(TOY)
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in ft.values())
    assert ft["lp"] <= ft["greedy"] and ft["greedy_n_open"] == 1
    assert ft["bound_nec"] == 50 and ft["bound_suf"] == 160      # without opt (cf. test_qubo)
    assert "opt" not in ft
