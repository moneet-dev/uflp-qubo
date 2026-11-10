"""MILP tests: the paper's toy example, and (M1 acceptance) all 37 published optima."""
import numpy as np
import pytest

from uflp.io import DATA_DIR, Instance, load_instance, load_optima
from uflp.milp import lp_bound, solution_cost, solve_uflp

TOY = Instance("toy", f=np.array([100.0, 125, 125]),
               c=np.array([[35.0, 40, 60, 60], [40, 55, 50, 60], [40, 30, 45, 50]]))

HAVE_DATA = (DATA_DIR / "optima.csv").exists()
OPTIMA = load_optima() if HAVE_DATA else []


def test_toy_optimum_is_290_with_facility_3():
    res = solve_uflp(TOY)
    assert res.cost == 290
    assert res.open_facilities == [2]
    assert res.assignment == [2, 2, 2, 2]


def test_lp_bound_is_a_lower_bound():
    assert lp_bound(TOY) <= 290 + 1e-9


def test_solution_cost_rejects_closed_facility():
    with pytest.raises(ValueError):
        solution_cost(TOY, [0], [0, 1, 0, 0])


@pytest.mark.data
@pytest.mark.skipif(not HAVE_DATA, reason="run data/download.py first")
@pytest.mark.parametrize("row", OPTIMA, ids=[r["name"] for r in OPTIMA])
def test_milp_matches_published_optimum(row):
    inst = load_instance(row["name"])
    res = solve_uflp(inst)
    assert res.cost == pytest.approx(row["opt"], rel=1e-6)
    assert lp_bound(inst) <= res.cost * (1 + 1e-9)
