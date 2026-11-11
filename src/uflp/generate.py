"""Seeded synthetic UFLP instances for the ML extension (training data; CLAUDE.md §11).

Four families give different cost structures and different LP gaps (the distance between the
LP-relaxation bound and the optimum, which [P] found decisive for solver difficulty):

* ``euclid``     sites and customers are random points in the unit square; c_ij = demand_j x
                 distance x 1,000; fixed costs vary by site.
* ``uniform``    c_ij drawn independently and uniformly; fixed costs vary by site. No geometry,
                 so the LP relaxation tends to be weaker.
* ``orlib_like`` Euclidean costs with one fixed cost shared by all sites, like the ORLIB cap files.
* ``gap``        each customer has only a few cheap sites (the rest are expensive) and all sites
                 cost the same to open, after the "large duality gap" construction of Kochetov &
                 Ivanenko (2005) used for class B-2 in [P]. Expensive costs are finite so the QUBO
                 stays well scaled.

``phi`` sets the fixed-to-service balance: the mean fixed cost is phi x the mean cheapest service
cost of one customer, times n / m. Small phi favours opening many sites; large phi, few sites.
``design`` draws phi from {0.5, 1, 2, 4, 8} for the first three families and from {8, 16, 32} for
``gap``, whose LP gap only appears when fixed costs are high.
Everything is a pure function of the seed, so instances are regenerated rather than stored.
"""
from __future__ import annotations

import numpy as np

from .io import Instance

FAMILIES = ("euclid", "uniform", "orlib_like", "gap")


def _fixed_costs(rng: np.random.Generator, m: int, n: int, c: np.ndarray, phi: float,
                 equal: bool) -> np.ndarray:
    """Fixed costs scaled so that phi controls how many sites are worth opening."""
    level = phi * c.min(axis=0).mean() * n / m
    if equal:
        return np.full(m, round(level, 2))
    return np.round(level * rng.uniform(0.5, 1.5, m), 2)


def generate(family: str, m: int, n: int, phi: float, seed: int) -> Instance:
    """Build one instance of the given family and size; identical inputs give identical instances."""
    rng = np.random.default_rng(seed)
    if family in ("euclid", "orlib_like"):
        sites, custs = rng.random((m, 2)), rng.random((n, 2))
        demand = rng.integers(5, 101, n)
        dist = np.linalg.norm(sites[:, None, :] - custs[None, :, :], axis=2)
        c = np.round(dist * demand[None, :] * 1000, 2)
    elif family == "uniform":
        c = np.round(rng.uniform(100, 5000, (m, n)), 2)
    elif family == "gap":
        # k cheap sites per customer, spread evenly over sites (each site is cheap for about
        # n*k/m customers), as in Kochetov & Ivanenko's construction. Tuned on 11 Nov 2026:
        # LP gaps of about 2-28%, exact MILP < 20 s at these sizes.
        k = 3 if m <= 30 else 5
        c = np.round(rng.uniform(5000, 10000, (m, n)), 2)
        order = np.concatenate([rng.permutation(m) for _ in range(-(-(n * k) // m) + 1)])
        pos = 0
        for j in range(n):
            cheap: list[int] = []
            while len(cheap) < k:
                i = int(order[pos])
                pos += 1
                if i not in cheap:
                    cheap.append(i)
            c[cheap, j] = np.round(rng.uniform(0, 400, k), 2)
    else:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    f = _fixed_costs(rng, m, n, c, phi, equal=family in ("orlib_like", "gap"))
    return Instance(name=f"{family}_m{m}_phi{phi:g}_s{seed}", f=f, c=c)


def design(n_instances: int = 200, seed: int = 2026) -> list[dict]:
    """The seeded list of instance specs: family, m, n = 50, phi and the instance seed."""
    rng = np.random.default_rng(seed)
    sizes = [16, 20, 25, 30, 40, 50]
    phis = [0.5, 1.0, 2.0, 4.0, 8.0]
    gap_phis = [8.0, 16.0, 32.0]                 # high fixed costs give the gap family its LP gap
    specs = []
    for k in range(n_instances):
        family = FAMILIES[k % len(FAMILIES)]
        specs.append({
            "family": family,
            "m": int(rng.choice(sizes)),
            "n": 50,
            "phi": float(rng.choice(gap_phis if family == "gap" else phis)),
            "seed": 10_000 + k,
        })
    return specs
