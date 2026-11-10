"""Download the 37 ORLIB instances used in this study and their *uncapacitated* optima.

Sources (see CLAUDE.md section 7):

* Instance files: OR-Library at Brunel (authoritative), for all 37 files.
* Uncapacitated optima for cap71-74, cap101-104, cap131-134: Brunel's ``uncapopt.txt``.
* Uncapacitated optima for cap41-44, cap51, cap61-64, cap81-84, cap91-94, cap111-114,
  cap121-124: UflLib's ORLIB package (``*.txt.opt`` files; the last number is the cost).
  UflLib (resources.mpi-inf.mpg.de) returned HTTP 403 on 9 Nov 2026 and the package is
  not in the Wayback Machine, so we read a GitHub copy of that package, pinned to one
  commit. We trust it only after three checks done by this script and by M1:
    1. its instance files are token-for-token identical to Brunel's;
    2. its optima for cap71-74/101-104/131-134 equal Brunel's ``uncapopt``;
    3. our own exact MILP solve (scripts/verify_milp_optima.py) reproduces every value.

Writes ``data/raw/<name>.txt`` and ``data/optima.csv`` (name, class, opt, source).
Run:  python data/download.py
"""
from __future__ import annotations

import csv
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
BRUNEL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files"
MIRROR_SHA = "a33e86443594f049bef9cea5776c64c797dc7f8c"
MIRROR = f"https://raw.githubusercontent.com/ShahSaudAbdali/uflp/{MIRROR_SHA}/ORLIB"

CLASSES: dict[str, list[str]] = {
    "A-1": ["cap41", "cap42", "cap43", "cap44", "cap51",
            "cap61", "cap62", "cap63", "cap64", "cap71", "cap72", "cap73", "cap74"],
    "A-2": ["cap81", "cap82", "cap83", "cap84", "cap91", "cap92", "cap93", "cap94",
            "cap101", "cap102", "cap103", "cap104"],
    "B-1": ["cap111", "cap112", "cap113", "cap114", "cap121", "cap122", "cap123", "cap124",
            "cap131", "cap132", "cap133", "cap134"],
}
UNCAP_SET = {f"cap{k}" for k in (71, 72, 73, 74, 101, 102, 103, 104, 131, 132, 133, 134)}


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


def mirror_dir(name: str) -> str:
    """Folder of an instance inside the UflLib package layout, e.g. 'ORLIB-cap/40'."""
    num = int(name[3:])
    group = num // 10 * 10
    kind = "ORLIB-uncap" if name in UNCAP_SET else "ORLIB-cap"
    return f"{kind}/{group}"


def parse_uncapopt(text: str) -> dict[str, float]:
    """Parse Brunel's uncapopt.txt: a header line, then 'name value' per line."""
    out: dict[str, float] = {}
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = float(parts[1])
    return out


def main() -> int:
    """Download everything, run the consistency checks, and write data/optima.csv."""
    RAW.mkdir(parents=True, exist_ok=True)
    brunel_opt = parse_uncapopt(fetch(f"{BRUNEL}/uncapopt.txt"))
    (RAW / "uncapopt.txt").write_text(fetch(f"{BRUNEL}/uncapopt.txt"))

    rows = []
    for cls, names in CLASSES.items():
        for name in names:
            brunel_txt = fetch(f"{BRUNEL}/{name}.txt")
            (RAW / f"{name}.txt").write_text(brunel_txt)
            mirror_txt = fetch(f"{MIRROR}/{mirror_dir(name)}/{name}.txt")
            if brunel_txt.split() != mirror_txt.split():
                raise SystemExit(f"{name}: mirror instance differs from Brunel's file")
            opt_file = fetch(f"{MIRROR}/{mirror_dir(name)}/{name}.txt.opt")
            (RAW / f"{name}.txt.opt").write_text(opt_file)
            mirror_opt = float(opt_file.split()[-1])
            if name in UNCAP_SET:
                if abs(mirror_opt - brunel_opt[name]) > 1e-6 * brunel_opt[name]:
                    raise SystemExit(f"{name}: mirror opt {mirror_opt} != Brunel {brunel_opt[name]}")
                opt, src = brunel_opt[name], "brunel_uncapopt"
            else:
                opt, src = mirror_opt, "ufllib_opt_file"
            rows.append({"name": name, "class": cls, "opt": opt, "source": src})
            print(f"{name:7s} {cls}  opt={opt:<14} ({src})")

    with open(ROOT / "optima.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "class", "opt", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} instances to {RAW} and {ROOT / 'optima.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
