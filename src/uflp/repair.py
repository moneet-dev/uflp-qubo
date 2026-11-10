"""Turn any annealer state into a feasible UFLP solution.

Repair keeps the facilities the state opened (y_i = 1) and sends every customer to its
cheapest open facility. The x part of the state is ignored. If the state opens no
facility, there is nothing to keep and repair fails (returns None); we do not invent a
facility, so a penalty that is too small shows up as failure rather than being hidden.
"""
from __future__ import annotations

import numpy as np

from .io import Instance


def repair_cost(inst: Instance, y: np.ndarray) -> float | None:
    """Cost of opening {i : y_i = 1} and assigning each customer to its cheapest open site."""
    opened = np.flatnonzero(np.asarray(y) > 0)
    if opened.size == 0:
        return None
    return float(inst.f[opened].sum() + inst.c[opened].min(axis=0).sum())


def repair_costs(inst: Instance, ys: np.ndarray) -> np.ndarray:
    """repair_cost for each row of ys (reads x m); NaN where no facility is open."""
    ys = np.atleast_2d(ys).astype(bool)
    fixed = ys @ inst.f
    big = np.where(ys[:, :, None], inst.c[None, :, :], np.inf)   # reads x m x n
    service = big.min(axis=1).sum(axis=1)
    out = fixed + service
    out[~ys.any(axis=1)] = np.nan
    return out
