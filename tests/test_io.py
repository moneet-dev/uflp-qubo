"""Parser tests: sizes, and that c[i][j] is facility i -> customer j (the file needs a transpose)."""
from pathlib import Path

import numpy as np
import pytest

from uflp.io import DATA_DIR, load_instance, load_optima, parse_orlib

# 2 facilities x 3 customers. Customer rows list (demand, cost from fac 1, cost from fac 2),
# deliberately wrapped across lines; facility 2 writes "capacity" instead of a number.
TINY = """2 3
 100 10.
 capacity 20.
 5
 1. 2.
 6 3. 4.
 7 5.
 6.
"""

HAVE_DATA = (DATA_DIR / "raw" / "cap71.txt").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="run data/download.py first")


def test_tiny_sizes_and_transpose():
    inst = parse_orlib(TINY, "tiny")
    assert (inst.m, inst.n) == (2, 3)
    np.testing.assert_array_equal(inst.f, [10.0, 20.0])
    # customer 1 costs (1, 2), customer 2 (3, 4), customer 3 (5, 6)
    np.testing.assert_array_equal(inst.c, [[1, 3, 5], [2, 4, 6]])


def test_trailing_tokens_rejected():
    with pytest.raises(ValueError):
        parse_orlib(TINY + " 99", "bad")


@needs_data
def test_cap71_orientation():
    inst = load_instance("cap71")
    assert (inst.m, inst.n) == (16, 50)
    assert inst.c.shape == (16, 50)
    assert inst.f[10] == 0.0 and inst.f[0] == 7500.0
    # first customer line of cap71.txt: "146" then 6739.725 10355.05 7650.4 ...
    assert inst.c[0, 0] == pytest.approx(6739.725)
    assert inst.c[1, 0] == pytest.approx(10355.05)
    assert inst.c[2, 0] == pytest.approx(7650.4)


@needs_data
@pytest.mark.parametrize("name,m,n", [("cap41", 16, 50), ("cap81", 25, 50), ("cap111", 50, 50)])
def test_sizes_per_class(name, m, n):
    inst = load_instance(name)
    assert (inst.m, inst.n) == (m, n)
    assert inst.m * inst.n + inst.m in (816, 1275, 2550)   # variable counts from [P Table 1]


@needs_data
def test_optima_table_has_37_rows():
    rows = load_optima()
    assert len(rows) == 37
    assert {r["class"] for r in rows} == {"A-1", "A-2", "B-1"}
    assert sum(r["class"] == "A-1" for r in rows) == 13


# UflLib "simple" format: 2 facilities x 3 customers; .opt numbers facilities from 0 and ends with the cost
SIMPLE = """FILE: tiny
2 3 0
1 10 1 2 30
2 12 8 7 1
"""


def test_ufllib_simple_parser_and_opt_reader():
    from uflp.io import parse_ufllib_simple, ufllib_opt_assignment
    from uflp.milp import solution_cost, solve_uflp
    inst = parse_ufllib_simple(SIMPLE, "tiny")
    assert (inst.m, inst.n) == (2, 3) and list(inst.f) == [10, 12]
    np.testing.assert_array_equal(inst.c, [[1, 2, 30], [8, 7, 1]])    # facility x customer, no transpose
    assign, cost = ufllib_opt_assignment("0 0 1 26", inst.n)
    assert assign == [0, 0, 1] and cost == 26
    assert solution_cost(inst, [0, 1], assign) == 26 == solve_uflp(inst).cost    # both open: 10+12+1+2+1
    with pytest.raises(ValueError):
        ufllib_opt_assignment("0 0", inst.n)


@pytest.mark.skipif(not (DATA_DIR / "raw" / "bk" / "B1.1").exists(), reason="run data/download_bk.py")
def test_bilde_krarup_b11_published_optimum():
    from uflp.io import parse_ufllib_simple, ufllib_opt_assignment
    from uflp.milp import solution_cost
    inst = parse_ufllib_simple((DATA_DIR / "raw" / "bk" / "B1.1").read_text(), "B1.1")
    assign, cost = ufllib_opt_assignment((DATA_DIR / "raw" / "bk" / "B1.1.opt").read_text(), inst.n)
    assert (inst.m, inst.n) == (50, 100) and cost == 23468
    assert solution_cost(inst, sorted(set(assign)), assign) == cost
