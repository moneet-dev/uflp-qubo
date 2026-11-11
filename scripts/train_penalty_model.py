"""M5: train the penalty model and compare it with fixed rules (CLAUDE.md §11).

Protocols
  A. leave-one-family-out: train on three generator families, test on the fourth. The held-out
     `gap` family (positive LP gap) is the headline test.
  B. stratified 5-fold cross-validation over all 200 instances (within-distribution).
  C. ORLIB sanity check: train on all 200 synthetic instances, apply to the 12 distinct ORLIB
     problems, using the M3 runs plus results/orlib_extra_*.csv. On ORLIB the LP bound equals the
     optimum, so this is a sanity check only.

Every fitted quantity (model, best fixed r, k) uses training instances only. Writes
results/ml_eval_<ts>.md and results/ml_predictions_<ts>.csv.

Run:  python scripts/train_penalty_model.py --instances results/ml_instances_<ts>.csv [more ...] \
          --runs results/ml_runs_<ts>.csv [more ...] \
          --m3-runs results/reproduction_<ts>.csv --orlib-extra results/orlib_extra_<ts>.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from uflp.features import features
from uflp.io import load_instance
from uflp.milp import lp_bound
from uflp.ml import (FEATURES, add_derived_features, fit_bound_k, fit_fixed_best, fit_model,
                     oracle_r, outcomes, predict_rules, snap, summarize)

ORLIB = [f"cap{k}" for k in (71, 72, 73, 74, 101, 102, 103, 104, 131, 132, 133, 134)]


def check_labels_match(runs: pd.DataFrame, r_star: pd.Series) -> None:
    """The label here must equal scripts/verify_labels.py's rule (mean cost, failures = inf)."""
    c = runs.assign(c=runs["repaired_cost"].fillna(np.inf))
    g = c.groupby(["instance", "r"]).agg(c=("c", "mean"), cpu=("cpu_time_s", "mean")).reset_index()
    ref = g.sort_values(["instance", "c", "cpu", "r"]).groupby("instance").head(1).set_index("instance")["r"]
    bad = (ref.loc[r_star.index] != r_star).sum()
    if bad:
        raise SystemExit(f"{bad} labels differ from verify_labels.py's rule")


def md(df: pd.DataFrame) -> str:
    """Small markdown table writer (no extra dependency)."""
    df = df.copy()
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def orlib_features() -> pd.DataFrame:
    """Features of the 12 distinct ORLIB problems (LP bound computed, never the optimum)."""
    rows = []
    for name in ORLIB:
        inst = load_instance(name)
        rows.append({"instance": name, **features(inst, lp=lp_bound(inst))})
    return add_derived_features(pd.DataFrame(rows).set_index("instance"))


def main() -> None:
    """Run protocols A-C and write the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=Path, nargs="+", required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--m3-runs", type=Path, required=True)
    ap.add_argument("--orlib-extra", type=Path, required=True)
    args = ap.parse_args()

    inst = add_derived_features(pd.concat([pd.read_csv(f) for f in args.instances]).set_index("instance"))
    runs = pd.concat([pd.read_csv(f) for f in args.runs], ignore_index=True)
    if inst.index.duplicated().any():
        raise SystemExit("duplicate instance names across instance files")
    out = outcomes(runs)
    r_star = oracle_r(out)
    check_labels_match(runs, r_star)
    X = inst.loc[out.gap.index]
    fam = X["family"]
    files = ", ".join(f"`{f.name}`" for f in args.instances + args.runs)
    text = [f"# M5 evaluation\n\nData: {files} (synthetic, {len(X)} instances, families "
            f"{sorted(fam.unique())}); `{args.m3_runs.name}` + `{args.orlib_extra.name}` (ORLIB). "
            "Gap = mean repaired gap over seeds "
            "(failed runs score 100%). CI = paired bootstrap 95% interval of (rule - model) over instances; "
            "positive means the rule is worse than the model.\n"]
    preds = []

    # A. leave-one-family-out
    text.append("## A. Leave-one-family-out (train on the other families, test on one)\n")
    for f in sorted(fam.unique(), key=lambda s: (s != "gap", s)):    # headline (gap) first
        train, test = list(X.index[fam != f]), list(X.index[fam == f])
        rules = predict_rules(out, X, train, test)
        text.append(f"### held-out family: `{f}` ({len(test)} instances)\n\n{md(summarize(out, rules))}\n")
        for name, r in rules.items():
            preds.append(pd.DataFrame({"instance": r.index, "protocol": f"LOFO:{f}", "rule": name, "r": r.values}))

    # B. stratified 5-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pooled: dict[str, list[pd.Series]] = {}
    for tr, te in skf.split(X, fam):
        rules = predict_rules(out, X, list(X.index[tr]), list(X.index[te]))
        for name, r in rules.items():
            key = "k x exactness bound (k tuned per fold)" if name.startswith("k x") else name
            pooled.setdefault(key, []).append(r)
    pooled_rules = {k: pd.concat(v) for k, v in pooled.items()}
    text.append(f"## B. Stratified 5-fold cross-validation (all {len(X)} instances)\n\n{md(summarize(out, pooled_rules))}\n")
    for name, r in pooled_rules.items():
        preds.append(pd.DataFrame({"instance": r.index, "protocol": "5fold", "rule": name, "r": r.values}))

    # C. ORLIB sanity check
    m3 = pd.read_csv(args.m3_runs)
    extra = pd.read_csv(args.orlib_extra)
    orl = pd.concat([m3[m3["instance"].isin(ORLIB)], extra], ignore_index=True)
    out_o = outcomes(orl)
    Xo = orlib_features().loc[out_o.gap.index]
    model = fit_model(X, r_star)
    k = fit_bound_k(out, X, list(X.index))
    fixed = fit_fixed_best(out, list(X.index))
    s = lambda v: pd.Series(snap(v, out_o.grid), index=Xo.index)  # noqa: E731
    rules_o = {
        "model (XGBoost)": s(np.exp(model.predict(Xo[FEATURES]))),
        "best fixed r (train)": s(np.full(len(Xo), fixed)),
        f"k x exactness bound (k={k:g})": s(k * Xo["bound_suf_rel"]),
        "LP/n (pilot rule)": s(1.0 / Xo["n"]),
        "50% of LP (paper)": s(np.full(len(Xo), 0.5)),
        "100% of LP (tutorial)": s(np.full(len(Xo), 1.0)),
        "oracle r*": oracle_r(out_o).loc[Xo.index],
    }
    text.append(f"## C. ORLIB sanity check (trained on all {len(X)} synthetic instances; 12 problems, 10 seeds)\n\n"
                f"{md(summarize(out_o, rules_o))}\n\nPer-problem r: "
                + ", ".join(f"{p}: model {rules_o['model (XGBoost)'][p]:g} / oracle {rules_o['oracle r*'][p]:g}"
                            for p in Xo.index) + "\n")
    for name, r in rules_o.items():
        preds.append(pd.DataFrame({"instance": r.index, "protocol": "ORLIB", "rule": name, "r": r.values}))

    # feature importance of the model trained on all synthetic instances
    imp = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(FEATURES).fillna(0)
    imp = (imp / imp.sum()).sort_values(ascending=False).round(3)
    text.append(f"## Feature importance (gain share, model trained on all {len(X)})\n\n"
                + md(imp.to_frame("gain share")) + "\n")

    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    res_dir = args.runs[0].parent
    (res_dir / f"ml_eval_{ts}.md").write_text("\n".join(text), encoding="utf-8")
    pd.concat(preds).to_csv(res_dir / f"ml_predictions_{ts}.csv", index=False)
    print("\n".join(text))
    print(f"\nWrote {res_dir / f'ml_eval_{ts}.md'} and ml_predictions_{ts}.csv")


if __name__ == "__main__":
    main()
