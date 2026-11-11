"""Cheap instance features for predicting a good penalty P (CLAUDE.md §11). None uses the optimum.

Costs are also given relative to the LP bound, so the features describe the *shape* of an
instance rather than its currency scale: the label r* = P*/LP is scale-free too.
"""
from __future__ import annotations

import numpy as np

from .heuristics import greedy_add
from .io import Instance
from .milp import lp_bound
from .qubo import penalty_bounds


def features(inst: Instance, lp: float | None = None) -> dict:
    """Feature dict for one instance; pass `lp` to reuse an LP bound already computed."""
    lp = lp_bound(inst) if lp is None else lp
    greedy, opened = greedy_add(inst)
    cheapest = inst.c.min(axis=0)                          # min_i c_ij per customer
    nec, suf = penalty_bounds(inst)                        # without opt: max_j min_i c_ij, sufficient bound
    return {
        "m": inst.m, "n": inst.n,
        "f_mean": float(inst.f.mean()), "f_std": float(inst.f.std()),
        "f_min": float(inst.f.min()), "f_max": float(inst.f.max()),
        "c_mean": float(inst.c.mean()), "c_std": float(inst.c.std()),
        "cheapest_mean": float(cheapest.mean()),
        "f_to_service": float(inst.f.mean() / cheapest.sum()),
        "lp": float(lp), "greedy": greedy,
        "greedy_lp_gap": float((greedy - lp) / lp),        # proxy for the LP (duality) gap
        "greedy_n_open": len(opened),
        "bound_nec": nec, "bound_suf": suf,
        # scale-free versions, relative to the LP bound
        "f_mean_rel": float(inst.f.mean() / lp), "cheapest_max_rel": float(cheapest.max() / lp),
        "bound_nec_rel": nec / lp, "bound_suf_rel": suf / lp, "inv_n": 1.0 / inst.n,
    }
