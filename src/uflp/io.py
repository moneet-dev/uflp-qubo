"""Read OR-Library / UflLib facility-location files and the optima table.

File format (OR-Library "cap" files):

    m n
    capacity_1 fixed_cost_1          (m lines, one per facility)
    ...
    demand_1                          (then, for each customer j:)
    c_1j c_2j ... c_mj                its demand, then the cost of serving all of
    ...                               j's demand from each facility i = 1..m

Capacities and demands are ignored: the *uncapacitated* facility location problem
(UFLP) only needs the fixed costs f_i and service costs c_ij. Because the file lists
costs customer by customer, we transpose so that ``c[i][j]`` is facility i -> customer j.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Instance:
    """A UFLP instance: open facilities (fixed cost f_i) and assign each customer j (cost c_ij)."""

    name: str
    f: np.ndarray  # shape (m,)
    c: np.ndarray  # shape (m, n); c[i, j] = cost for facility i to serve customer j

    @property
    def m(self) -> int:
        """Number of candidate facility sites."""
        return int(self.f.shape[0])

    @property
    def n(self) -> int:
        """Number of customers."""
        return int(self.c.shape[1])


def parse_orlib(text: str, name: str = "") -> Instance:
    """Parse ORLIB text as a whitespace token stream (lines wrap) into an Instance."""
    toks = text.split()
    pos = 0

    def take() -> str:
        nonlocal pos
        tok = toks[pos]
        pos += 1
        return tok

    m, n = int(take()), int(take())
    f = np.empty(m)
    for i in range(m):
        take()  # capacity, ignored (capa-c write the word "capacity"; it is still one token)
        f[i] = float(take())
    c_t = np.empty((n, m))  # as stored in the file: one row per customer
    for j in range(n):
        take()  # demand, ignored
        for i in range(m):
            c_t[j, i] = float(take())
    if pos != len(toks):
        raise ValueError(f"{name}: {len(toks) - pos} unread tokens; file layout not as expected")
    return Instance(name=name, f=f, c=c_t.T.copy())


def load_instance(name: str, raw_dir: Path | None = None) -> Instance:
    """Load data/raw/<name>.txt."""
    path = (raw_dir or DATA_DIR / "raw") / f"{name}.txt"
    return parse_orlib(path.read_text(), name=name)


def load_optima(path: Path | None = None) -> list[dict]:
    """Read data/optima.csv (written by data/download.py): name, class, opt, source."""
    with open(path or DATA_DIR / "optima.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["opt"] = float(r["opt"])
    return rows
