"""Check and summarize the capacitated sweep, and compare it with the uncapacitated M3 sweep.

Checks: completeness of the (problem, k, r, seed) grid; no feasible or repaired cost below the
CFLP optimum; CFLP optimum >= UFLP optimum; a sample of runs re-executed gives identical results.
Comparison: the same 12 problems, r values and seeds 0-4 from results/reproduction_*.csv
(uncapacitated, canonical files cap7x/10x/13x).

Writes results/capacitated_analysis_<ts>.md and results/figures/capacitated_gap.png.
Run:  python scripts/analyze_capacitated.py results/capacitated_<ts>.csv results/reproduction_<ts>.csv
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_capacitated as rc  # noqa: E402

COLORS = {"uncapacitated": "#2a78d6", "k = 10": "#eb6834", "k = 5": "#1baf7a"}   # validated slots 1-3
MARKERS = {"uncapacitated": "o", "k = 10": "s", "k = 5": "^"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def per_r(df: pd.DataFrame, col: str) -> pd.Series:
    """Mean over seeds per problem, then over problems, for each r."""
    return df.groupby(["problem", "r"])[col].mean().groupby("r").mean()


def main() -> int:
    """Run the checks, write tables and the figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cap_csv", type=Path)
    ap.add_argument("m3_csv", type=Path)
    ap.add_argument("--sample", type=int, default=6)
    args = ap.parse_args()
    cap = pd.read_csv(args.cap_csv)
    meta = json.loads(args.cap_csv.with_suffix(".json").read_text())
    lines, ok_all = [], True

    def report(name: str, ok: bool, detail: str) -> None:
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        print(lines[-1], flush=True)

    expect = {(f"{p}_k{k}", r, s) for p in meta["problems"] for k in meta["k"]
              for r in meta["r_grid"] for s in meta["seeds"]}
    got = list(zip(cap["instance"], cap["r"], cap["seed"]))
    report("completeness", set(got) == expect and len(got) == len(expect), f"{len(got)} rows, {len(expect)} expected")
    tol = 1e-9 * cap["opt"]
    rep_ok = (cap["repaired_cost"].dropna() >= (cap["opt"] - tol)[cap["repaired_cost"].notna()]).all()
    raw_ok = (cap["raw_best_cost"].dropna() >= (cap["opt"] - tol)[cap["raw_best_cost"].notna()]).all()
    cu_ok = (cap["opt"] >= cap["opt_uflp"] * (1 - 1e-12)).all()
    report("invariants", rep_ok and raw_ok and cu_ok,
           f"repaired >= CFLP opt {rep_ok}, raw feasible >= CFLP opt {raw_ok}, CFLP opt >= UFLP opt {cu_ok}")
    rng = random.Random(5)
    mism = 0
    for x in rng.sample(cap.to_dict("records"), args.sample):
        spec = {k: x[k] for k in ["instance", "problem", "class", "k", "opt", "opt_uflp", "binding", "r", "seed"]}
        spec["formulation"] = x.get("formulation") if isinstance(x.get("formulation"), str) else "coupled"
        again = rc.job(spec)
        fields = ["best_energy", "n_feasible_reads", "n_slack_exact_reads", "raw_best_cost", "repaired_cost", "n_repair_fail"]
        mism += not all((pd.isna(x[f]) and pd.isna(again[f])) or x[f] == again[f] for f in fields)
    report("reproducible", mism == 0, f"{args.sample} sampled runs re-executed, {mism} differ")

    m3 = pd.read_csv(args.m3_csv)
    unc = m3[m3["instance"].isin(meta["problems"]) & m3["seed"].isin(meta["seeds"])].copy()
    unc["problem"] = unc["instance"]
    groups = {"uncapacitated": unc, "k = 10": cap[cap["k"] == 10], "k = 5": cap[cap["k"] == 5]}
    for g in groups.values():
        g["feas_pct"] = 100 * g["n_feasible_reads"] / g["reads"]
        g["gap_eval"] = g["gap_pct"].fillna(100.0)          # a run with no repairable read scores 100%

    tab = pd.DataFrame({name: per_r(g, "gap_eval") for name, g in groups.items()}).round(2)
    feas = pd.DataFrame({name: per_r(g, "feas_pct") for name, g in groups.items()}).round(1)
    raw = pd.DataFrame({name: per_r(g, "raw_gap_pct") for name, g in groups.items()}).round(2)
    nvars = cap.groupby("k")["n_vars"].agg(["min", "max"])
    exact = cap.assign(e=100 * cap["n_slack_exact_reads"] / cap["reads"]).groupby(["k", "r"])["e"].mean().unstack("k").round(1)
    fails = cap.groupby(["k", "r"])["n_repair_fail"].sum().unstack("k")
    best_r = {name: float(tab[name].idxmin()) for name in tab.columns}
    bind = cap.drop_duplicates("instance").groupby("k")["binding"].sum()

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for name in tab.columns:
        s = tab[name]
        ax.plot(s.index, s.values, color=COLORS[name], lw=2, marker=MARKERS[name], ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        ax.annotate(name, (s.index[-1], s.values[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.0085, 1.6)
    ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["1%", "2%", "5%", "10%", "25%", "50%", "100%"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("penalty P as % of the optimal cost (r)", color=INK, fontsize=9)
    ax.set_ylabel("gap after repair (%)", color=INK, fontsize=9)
    ax.set_title("SA gap vs penalty: uncapacitated vs at most k customers per facility",
                 color=INK, fontsize=10, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(0.93, 0.0))
    fig.tight_layout()
    figs = args.cap_csv.parent / "figures"
    figs.mkdir(exist_ok=True)
    fig.savefig(figs / f"{args.cap_csv.stem.rsplit('_', 2)[0]}_gap.png", facecolor=SURFACE)
    plt.close(fig)

    def md(df: pd.DataFrame) -> str:
        head = "| r | " + " | ".join(map(str, df.columns)) + " |\n|---|" + "---|" * len(df.columns) + "\n"
        return head + "\n".join(f"| {r:g} | " + " | ".join("–" if pd.isna(v) else f"{v:g}" for v in row) + " |"
                                for r, row in df.iterrows())

    text = [f"# Capacitated extension: analysis of `{args.cap_csv.name}`\n",
            "12 distinct ORLIB problems; capacity of [UL] (A.4), at most k customers per facility. "
            f"Binding (CFLP opt > UFLP opt): k=5 {int(bind.get(5, 0))}/12, k=10 {int(bind.get(10, 0))}/12. "
            f"QUBO variables: k=5 {int(nvars.loc[5, 'min'])}–{int(nvars.loc[5, 'max'])}, "
            f"k=10 {int(nvars.loc[10, 'min'])}–{int(nvars.loc[10, 'max'])} (uncapacitated 816–2,550). "
            "Uncapacitated reference: the same problems, r and seeds 0-4 from "
            f"`{args.m3_csv.name}`. Averages are over problems.\n",
            "## Checks\n", "\n".join(lines) + "\n",
            "## Mean repaired gap % (a run with no repairable read scores 100%)\n", md(tab) + "\n",
            f"Best r: {best_r}\n",
            "## Raw-feasible reads % (original constraints satisfied)\n", md(feas) + "\n",
            "## Reads that are feasible *and* have exact slack (energy = cost), %\n", md(exact) + "\n",
            "## Mean raw gap % (best feasible read; – = none)\n", md(raw) + "\n",
            "## Reads where repair failed (some customer did not fit), total\n", md(fails) + "\n"]
    out = args.cap_csv.parent / f"capacitated_analysis_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text[4:]))
    print(f"\n{'ALL CHECKS PASS' if ok_all else 'SOME CHECKS FAILED'}; wrote {out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
