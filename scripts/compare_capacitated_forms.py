"""Compare the two capacity-row forms of the CFLP QUBO under SA (paired on the shared seeds).

coupled   : P * (sum_j x_ij + s_i - k y_i)^2   ([UL] (A.4) as written)
decoupled : P * (sum_j x_ij + s_i - k)^2       (x <= y is enforced by eq. 15's own penalty)

Same 24 problems (12 ORLIB cost matrices x k in {5, 10}), same r grid, same SA settings; the
comparison uses the seeds both sweeps share. The uncapacitated M3 runs on the same problems and
seeds are shown for reference.

Writes results/capacitated_forms_<ts>.md and results/figures/capacitated_forms.png.
Run:  python scripts/compare_capacitated_forms.py results/capacitated_<ts>.csv \
          results/capacitated_decoupled_<ts>.csv results/reproduction_<ts>.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from uflp.ml import bootstrap_ci  # noqa: E402

COLORS = {"uncapacitated": "#2a78d6", 10: "#eb6834", 5: "#1baf7a"}     # validated slots 1-3
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def main() -> None:
    """Tables, paired CIs and the figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("coupled", type=Path)
    ap.add_argument("decoupled", type=Path)
    ap.add_argument("m3", type=Path)
    args = ap.parse_args()
    co, de = pd.read_csv(args.coupled), pd.read_csv(args.decoupled)
    seeds = sorted(set(co["seed"]) & set(de["seed"]))
    co, de = co[co["seed"].isin(seeds)], de[de["seed"].isin(seeds)]
    for d in (co, de):
        d["gap_eval"] = d["gap_pct"].fillna(100.0)
        d["feas_pct"] = 100 * d["n_feasible_reads"] / d["reads"]
    m3 = pd.read_csv(args.m3)
    unc = m3[m3["instance"].isin(co["problem"].unique()) & m3["seed"].isin(seeds)]

    # per (problem, k, r): mean over seeds
    g = lambda d, col: d.groupby(["k", "problem", "r"])[col].mean()  # noqa: E731
    gc, gd = g(co, "gap_eval"), g(de, "gap_eval")
    rows = []
    for k in sorted(co["k"].unique()):
        for r in sorted(co["r"].unique()):
            a, b = gc.loc[k, :, r], gd.loc[k, :, r]
            diff = (a - b.loc[a.index]).to_numpy()
            lo, hi = bootstrap_ci(diff)
            rows.append({"k": k, "r": r, "coupled gap %": a.mean(), "decoupled gap %": b.mean(),
                         "coupled - decoupled (95% CI)": f"{diff.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]",
                         "coupled feasible reads %": co[(co.k == k) & (co.r == r)]["feas_pct"].mean(),
                         "decoupled feasible reads %": de[(de.k == k) & (de.r == r)]["feas_pct"].mean()})
    tab = pd.DataFrame(rows)
    unc_gap = unc.groupby(["instance", "r"])["gap_pct"].mean().groupby("r").mean()

    best = {}
    for k in sorted(co["k"].unique()):
        for name, gg in [("coupled", gc), ("decoupled", gd)]:
            s = gg.loc[k].groupby("r").mean()
            best[(k, name)] = (float(s.idxmin()), float(s.min()))

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.plot(unc_gap.index, unc_gap.values, color=COLORS["uncapacitated"], lw=2, marker="o", ms=5,
            markeredgecolor=SURFACE, label="uncapacitated")
    for k in sorted(co["k"].unique(), reverse=True):
        t = tab[tab.k == k].set_index("r")
        ax.plot(t.index, t["coupled gap %"], color=COLORS[k], lw=2, ls="--", marker="s", ms=5,
                markeredgecolor=SURFACE, label=f"k = {k}, coupled")
        ax.plot(t.index, t["decoupled gap %"], color=COLORS[k], lw=2, ls="-", marker="^", ms=5,
                markeredgecolor=SURFACE, label=f"k = {k}, decoupled")
    ax.set_xscale("log")
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
    ax.set_title("Capacity row with y (coupled, dashed) vs without y (decoupled, solid)",
                 color=INK, fontsize=10, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=7.5, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    figs = args.coupled.parent / "figures"
    figs.mkdir(exist_ok=True)
    fig.savefig(figs / "capacitated_forms.png", facecolor=SURFACE)
    plt.close(fig)

    def md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, row in df.iterrows():
            out.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + " |")
        return "\n".join(out)

    text = [f"# Capacity-row forms compared (`{args.coupled.name}` vs `{args.decoupled.name}`)\n",
            f"Seeds {seeds}; 12 ORLIB problems per k; mean repaired gap over problems (failed runs 100%). "
            "CI = paired bootstrap over the 12 problems.\n",
            md(tab) + "\n",
            "## Best r per form\n",
            "\n".join(f"- k = {k}, {name}: r = {r:g}, mean gap {v:.2f}%" for (k, name), (r, v) in best.items()) + "\n",
            "## Uncapacitated reference (same problems and seeds)\n",
            "\n".join(f"- r = {r:g}: {v:.2f}%" for r, v in unc_gap.items()) + "\n"]
    out = args.coupled.parent / f"capacitated_forms_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
