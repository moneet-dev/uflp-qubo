"""Do probe features fix the penalty model's transfer failure? Every model variant is trained with the
cost features only and with probe features added, and evaluated on:

  1. stratified 5-fold CV on the synthetic instances;
  2. the fresh Bilde-Krarup test set (pre-registered in scripts/run_bk.py; headline test);
  3. ORLIB (no longer untouched: the heavy_demand family was added after seeing it).

Models are trained on the synthetic instances only. Rows ending in "+ probes" use the cost features
plus src/uflp/probe.py's features. CI = paired bootstrap of (rule - cost-feature XGBoost model).

Writes results/probe_eval_<ts>.md. Run:
    python scripts/evaluate_probes.py --instances <ml_instances ...> --runs <ml_runs ...> \
        --bk-instances <bk_instances> --bk-runs <bk_runs> --m3-runs <reproduction> \
        --orlib-extra <orlib_extra> --probes <probe_features>
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_penalty_model import ORLIB, md, orlib_features  # noqa: E402

from uflp.ml import (FEATURES, add_derived_features, oracle_r, outcomes, predict_all, subset,  # noqa: E402
                     summarize)
from uflp.probe import PROBE_FEATURES  # noqa: E402

MODEL_RULES = ["model (XGBoost)", "model, rounded up one step", "gap-curve model", "linear model (ridge)"]


def both(out_tr, X_tr, X_te, grid) -> dict[str, pd.Series]:
    """All rules with cost features, plus the four model variants with probe features added."""
    rules = predict_all(out_tr, X_tr, X_te, grid, FEATURES)
    probe = predict_all(out_tr, X_tr, X_te, grid, FEATURES + PROBE_FEATURES)
    for name in MODEL_RULES:
        rules[f"{name} + probes"] = probe[name]
    return rules


def main() -> None:
    """Run the three evaluations and write the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=Path, nargs="+", required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--bk-instances", type=Path, required=True)
    ap.add_argument("--bk-runs", type=Path, required=True)
    ap.add_argument("--m3-runs", type=Path, required=True)
    ap.add_argument("--orlib-extra", type=Path, required=True)
    ap.add_argument("--probes", type=Path, required=True)
    args = ap.parse_args()
    probes = pd.read_csv(args.probes).set_index("instance")[PROBE_FEATURES]

    syn = add_derived_features(pd.concat([pd.read_csv(f) for f in args.instances]).set_index("instance"))
    out = outcomes(pd.concat([pd.read_csv(f) for f in args.runs], ignore_index=True))
    X = syn.loc[out.gap.index].join(probes)
    bk = add_derived_features(pd.read_csv(args.bk_instances).set_index("instance"))
    out_bk = outcomes(pd.read_csv(args.bk_runs))
    Xbk = bk.loc[out_bk.gap.index].join(probes)
    m3 = pd.read_csv(args.m3_runs)
    out_o = outcomes(pd.concat([m3[m3["instance"].isin(ORLIB)], pd.read_csv(args.orlib_extra)], ignore_index=True))
    Xo = orlib_features().loc[out_o.gap.index].join(probes)
    for name, df in [("synthetic", X), ("Bilde-Krarup", Xbk), ("ORLIB", Xo)]:
        if df[PROBE_FEATURES].isna().any().any():
            raise SystemExit(f"missing probe features for some {name} instances")

    text = [f"# Probe features: evaluation\n\nSynthetic: {len(X)} instances; Bilde-Krarup: {len(Xbk)} "
            f"(LP gap {bk['lp_gap_pct'].min():.1f}-{bk['lp_gap_pct'].max():.1f}%); ORLIB: {len(Xo)}. "
            "Mean repaired gap over instances (failed runs score 100%).\n"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pooled: dict[str, list[pd.Series]] = {}
    for tr, te in skf.split(X, X["family"]):
        trn, tst = list(X.index[tr]), list(X.index[te])
        rules = both(subset(out, trn), X.loc[trn], X.loc[tst], out.grid)
        rules["oracle r*"] = oracle_r(subset(out, tst)).loc[tst]
        for name, r in rules.items():
            pooled.setdefault("k x exactness bound (k tuned per fold)" if name.startswith("k x") else name, []).append(r)
    text.append("## 1. Stratified 5-fold CV on synthetic instances\n\n"
                + md(summarize(out, {k: pd.concat(v) for k, v in pooled.items()})) + "\n")

    for title, out_te, X_te in [("2. Fresh test set: Bilde-Krarup (trained on all synthetic)", out_bk, Xbk),
                                ("3. ORLIB (trained on all synthetic)", out_o, Xo)]:
        rules = both(out, X, X_te, out_te.grid)
        rules["oracle r*"] = oracle_r(out_te).loc[X_te.index]
        text.append(f"## {title}\n\n{md(summarize(out_te, rules))}\n")
        if title.startswith("2"):
            per = pd.DataFrame({"family": X_te["family"], **{n: rules[n] for n in
                                ["model (XGBoost)", "model (XGBoost) + probes", "oracle r*"]}})
            text.append("Predicted r by Bilde-Krarup set (median): "
                        + per.groupby("family").median(numeric_only=True).round(3).to_string().replace("\n", "; ") + "\n")

    out_path = args.runs[0].parent / f"probe_eval_{datetime.now():%Y%m%d_%H%M%S}.md"
    out_path.write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
