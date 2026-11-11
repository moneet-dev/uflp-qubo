"""Greedy UFLP heuristic: a quick upper bound used as an ML feature (no optimum needed).

Greedy "add": start with the single best facility, then keep opening the facility that lowers
total cost the most (its fixed cost against the service costs it saves), and stop when no
facility helps. Fast but not optimal; the gap between its cost and the LP lower bound hints at
how hard an instance is.
"""
from __future__ import annotations

import numpy as np

from .io import Instance


def greedy_add(inst: Instance) -> tuple[float, list[int]]:
    """Return (cost, opened facilities) of the greedy-add heuristic."""
    f, c = inst.f, inst.c
    first = int(np.argmin(f + c.sum(axis=1)))
    opened = [first]
    best_service = c[first].copy()                 # cheapest service cost so far, per customer
    cost = float(f[first] + best_service.sum())
    while True:
        savings = np.maximum(best_service[None, :] - c, 0).sum(axis=1) - f
        savings[opened] = -np.inf
        i = int(np.argmax(savings))
        if savings[i] <= 1e-9:
            break
        opened.append(i)
        best_service = np.minimum(best_service, c[i])
        cost -= float(savings[i])
    cost = float(f[opened].sum() + c[opened].min(axis=0).sum())   # recompute exactly
    return cost, sorted(opened)
