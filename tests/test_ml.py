"""M5 helper tests on a tiny hand-made table of SA outcomes."""
import numpy as np
import pandas as pd
import pytest

from uflp.ml import FAIL_GAP, bootstrap_ci, fit_fixed_best, lookup, oracle_r, outcomes, snap


def runs_table() -> pd.DataFrame:
    """Two instances x three r x two seeds; instance b fails once at r=0.01."""
    rows = []
    gaps = {("a", 0.01): [5, 7], ("a", 0.03): [1, 1], ("a", 0.1): [1, 1],
            ("b", 0.01): [0.5, None], ("b", 0.03): [4, 4], ("b", 0.1): [2, 2]}
    cpu = {0.01: 1.0, 0.03: 2.0, 0.1: 1.5}
    for (inst, r), gs in gaps.items():
        for seed, g in enumerate(gs):
            rows.append({"instance": inst, "r": r, "seed": seed, "gap_pct": g,
                         "repaired_cost": None if g is None else 100 + g,
                         "n_feasible_reads": 10, "reads": 50, "cpu_time_s": cpu[r]})
    return pd.DataFrame(rows)


def test_failed_run_scores_fail_gap_and_cannot_win():
    out = outcomes(runs_table())
    assert out.gap.at["b", 0.01] == (0.5 + FAIL_GAP) / 2
    assert out.fail.at["b", 0.01] == 0.5
    r = oracle_r(out)
    assert r["b"] == 0.1                 # 0.01 would win if the failure were dropped
    assert r["a"] == 0.1                 # tie on gap with 0.03: lower CPU wins


def test_snap_is_nearest_in_log_space():
    grid = np.array([0.01, 0.03, 0.1])
    assert list(snap([0.012, 0.02, 0.05, 0.06, 5.0], grid)) == [0.01, 0.03, 0.03, 0.1, 0.1]


def test_lookup_and_fixed_best():
    out = outcomes(runs_table())
    res = lookup(out, pd.Series({"a": 0.03, "b": 0.1}))
    assert list(res["gap"]) == [1, 2]
    assert fit_fixed_best(out, ["a", "b"]) == 0.1      # means: 0.01 -> 28.1, 0.03 -> 2.5, 0.1 -> 1.5


def test_bootstrap_ci_is_seeded_and_brackets_mean():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    assert bootstrap_ci(d) == bootstrap_ci(d)
    lo, hi = bootstrap_ci(d)
    assert lo <= d.mean() <= hi
    assert bootstrap_ci(np.zeros(5)) == pytest.approx((0.0, 0.0))


def test_step_up_moves_to_next_grid_value():
    from uflp.ml import step_up
    grid = np.array([0.01, 0.03, 0.1])
    assert list(step_up(np.array([0.01, 0.03, 0.1]), grid)) == [0.03, 0.1, 0.1]


def test_probe_features_are_seeded_and_need_no_optimum():
    from uflp.generate import generate
    from uflp.heuristics import greedy_add
    from uflp.milp import lp_bound
    from uflp.probe import PROBE_FEATURES, probe_features
    inst = generate("euclid", 16, 50, 2.0, seed=3)
    lp, (greedy, _) = lp_bound(inst), greedy_add(inst)
    a, b = probe_features(inst, lp, greedy), probe_features(inst, lp, greedy)
    assert a == b and set(a) == set(PROBE_FEATURES)          # consumers select columns by name
    assert all(0 <= a[f"feas_{r:g}"] <= 1 for r in (0.02, 0.05, 0.15))
    assert a["slope_hi"] == pytest.approx(a["rep_0.15"] - a["rep_0.05"])
