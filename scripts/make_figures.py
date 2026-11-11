"""M3 figures and tables: gap, feasibility and time vs r, one line per class (A-1, A-2, B-1).

Averaging: first over the files and seeds of each distinct problem (CLAUDE.md §7.1: the 37
files are 12 problems), then over the problems of a class, so duplicated problems do not
count extra. Paper values ([P] Tables 2-3, penalty-function approach) are shown in the tables
for the four r values the paper used.

Writes results/figures/m3_*.png and results/m3_tables_<timestamp>.md.
Run:  python scripts/make_figures.py results/reproduction_<ts>.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from uflp.io import load_instance  # noqa: E402
from uflp.qubo import penalty_bounds  # noqa: E402

CLASSES = ["A-1", "A-2", "B-1"]
COLORS = {"A-1": "#2a78d6", "A-2": "#eb6834", "B-1": "#1baf7a"}   # validated categorical slots 1-3
MARKERS = {"A-1": "o", "A-2": "s", "B-1": "^"}                      # secondary encoding
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
PAPER_GAP = {c: {0.25: 0.0, 0.5: 0.0, 0.75: 0.0, 1.0: 0.0} for c in CLASSES}      # [P] Table 2
PAPER_TIME = {"A-1": {0.25: 1.2, 0.5: 1.45, 0.75: 1.23, 1.0: 1.3},               # [P] Table 3
              "A-2": {0.25: 2.75, 0.5: 4.15, 0.75: 3.73, 1.0: 2.12},
              "B-1": {0.25: 18.09, 0.5: 1.73, 0.75: 1.79, 1.0: 1.78}}


def class_means(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean of `col` per (class, r): averaged within each problem first, then across problems."""
    per_problem = df.groupby(["class", "problem", "r"])[col].mean()
    return per_problem.groupby(["class", "r"]).mean().unstack("class")[CLASSES]


def style(ax, ylabel: str, title: str) -> None:
    """Recessive axes and grid, log r axis, paper's grid shaded."""
    ax.set_xscale("log")
    ax.set_xlim(0.0085, 1.3)                       # same r range on every figure
    ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["1%", "2%", "5%", "10%", "25%", "50%", "100%"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axvspan(0.25, 1.0, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.text(0.5, 1.0, "paper's P range", transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", fontsize=8, color=MUTED)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("penalty P as % of the optimal cost (r)", color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=14)
    ax.set_facecolor(SURFACE)


def plot(table: pd.DataFrame, ylabel: str, title: str, path: Path, label_last: bool = True) -> None:
    """One line per class with markers and direct end labels, plus a legend."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for c in CLASSES:
        s = table[c].dropna()
        ax.plot(s.index, s.values, color=COLORS[c], lw=2, marker=MARKERS[c], ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=c)
        if label_last and len(s):
            ax.annotate(c, (s.index[-1], s.values[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
    style(ax, ylabel, title)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def md_table(t: pd.DataFrame, fmt: str, paper: dict | None = None) -> str:
    """Markdown table of class x r, with the paper's value in brackets where it exists."""
    head = "| r | " + " | ".join(CLASSES) + " |\n|---|" + "---|" * len(CLASSES) + "\n"
    body = ""
    for r, row in t.iterrows():
        cells = []
        for c in CLASSES:
            v = "–" if pd.isna(row[c]) else format(row[c], fmt)
            if paper and r in paper[c]:
                v += f" [P: {paper[c][r]:g}]"
            cells.append(v)
        body += f"| {r:g} | " + " | ".join(cells) + " |\n"
    return head + body


def main() -> None:
    """Write the four figures and the markdown tables."""
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df["feas_rate"] = 100 * df["n_feasible_reads"] / df["reads"]
    figs = args.csv.parent / "figures"
    figs.mkdir(exist_ok=True)

    tables = {
        "repaired gap %": (class_means(df, "gap_pct"), ".2f", PAPER_GAP),
        "raw gap % (best raw-feasible read; – = no feasible read)": (class_means(df, "raw_gap_pct"), ".2f", PAPER_GAP),
        "raw-feasible reads %": (class_means(df, "feas_rate"), ".1f", None),
        "CPU time per run, s (paper: Gurobi time to best solution)": (class_means(df, "cpu_time_s"), ".2f", PAPER_TIME),
    }   # wall time (time_s) is not tabulated: runs that span a machine sleep inflate it; CPU time is robust
    plot(tables["repaired gap %"][0], "gap after repair (%)",
         "SA gap after repair vs penalty (paper's Gurobi QUBO: 0% at 25-100%)", figs / "m3_gap_repaired.png")
    plot(tables["raw gap % (best raw-feasible read; – = no feasible read)"][0], "gap of best feasible read (%)",
         "SA raw gap vs penalty (no repair, as in the paper; no feasible read below 25%)",
         figs / "m3_gap_raw.png")
    plot(tables["raw-feasible reads %"][0], "reads that are feasible (%)",
         "Feasibility of SA reads vs penalty (classes overlap)", figs / "m3_feasibility.png",
         label_last=False)
    plot(tables["CPU time per run, s (paper: Gurobi time to best solution)"][0], "CPU seconds per run",
         "SA CPU time per run (50 reads x 1,000 sweeps) vs penalty", figs / "m3_time.png")

    bounds = []
    for (cls, prob), _ in df.groupby(["class", "problem"]):
        inst = load_instance(prob)
        opt = df.loc[df["problem"] == prob, "opt"].iloc[0]
        nec, suf = penalty_bounds(inst, opt)
        bounds.append(f"| {prob} | {cls} | {nec / opt:.4f} | {suf / opt:.4f} |")

    out = args.csv.parent / f"m3_tables_{datetime.now():%Y%m%d_%H%M%S}.md"
    text = [f"# M3 tables from `{args.csv.name}`\n",
            "Class means over distinct problems (each problem averaged over its files and seeds first). "
            "`[P: x]` = the paper's penalty-function value (Tables 2-3, Gurobi QUBO solver).\n"]
    for name, (t, fmt, paper) in tables.items():
        text.append(f"## {name}\n\n{md_table(t, fmt, paper)}")
    text.append("## Exactness bounds on r = P/opt (qubo.penalty_bounds)\n\n| problem | class | necessary r >= | sufficient r > |\n|---|---|---|---|\n"
                + "\n".join(bounds) + "\n")
    out.write_text("\n".join(text), encoding="utf-8")
    print(f"wrote {figs}/m3_*.png and {out}")


if __name__ == "__main__":
    main()
