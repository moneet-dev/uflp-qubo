"""Download UflLib's Bilde-Krarup instances (sets B, C, Dq/1, Eq/1: 40 instances) as a fresh test set.

Bilde-Krarup-type problems for the uncapacitated facility location problem, from M. Hoefer's UflLib
(2002). UflLib itself was unreachable in November 2026, so this reads a GitHub copy of the package
(MarkoMAT4/RI2025-2026, pinned to one commit). Each instance comes with an .opt file giving the
optimal facility for every customer; scripts/verify_bk_optima.py checks those assignments against
our own exact MILP solve before the instances are used.

File format ("simple" UflLib format): line 1 "FILE: name"; line 2 "m n 0"; then one line per
facility: index, opening cost, then the cost of serving each of the n customers.

Writes data/raw/bk/<name> and data/raw/bk/<name>.opt. Run:  python data/download_bk.py
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw" / "bk"
SHA = "ac5a046b12581da85f69901967bc4485cb43ec54"
BASE = f"https://raw.githubusercontent.com/MarkoMAT4/RI2025-2026/{SHA}/BildeKrarup"
SETS = {"B": "B/B1.{k}", "C": "C/C1.{k}", "D": "Dq/1/D1.{k}", "E": "Eq/1/E1.{k}"}


def fetch(url: str, retries: int = 3) -> str:
    """Download a URL as text, retrying on transient network errors."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "uflp-qubo-research/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("ascii")
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> int:
    """Download the 40 instances and their optimal assignments."""
    RAW.mkdir(parents=True, exist_ok=True)
    for pattern in SETS.values():
        for k in range(1, 11):
            path = pattern.format(k=k)
            name = path.split("/")[-1]
            (RAW / name).write_text(fetch(f"{BASE}/{path}"))
            (RAW / f"{name}.opt").write_text(fetch(f"{BASE}/{path}.opt"))
            print(name, flush=True)
    print(f"wrote 40 instances to {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
