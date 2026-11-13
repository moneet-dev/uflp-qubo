"""M5: learn a penalty rule P = r x LP bound from cheap instance features, and evaluate rules.

Evaluation needs no new SA runs: every rule's r is snapped to the label grid (the nearest grid
value in log space), and its outcome is looked up from the SA runs already made at that r. A run
in which no read opened a facility has no solution; it scores FAIL_GAP (100%) and is counted as a
failure, so a rule cannot win by sometimes failing.

The model is a gradient-boosted tree regressor (XGBoost) on log r*: trees split the feature space
into regions and average the labels in each, and boosting adds small trees that each correct the
remaining error. Only scale-free features are used, so a rule learnt on synthetic costs can be
applied to ORLIB costs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

FAIL_GAP = 100.0
FEATURES = ["m", "f_to_service", "greedy_lp_gap", "open_frac", "f_mean_rel", "cheapest_max_rel",
            "bound_nec_rel", "bound_suf_rel", "f_cv", "c_cv"]
XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, random_state=0, n_jobs=1)


def add_derived_features(inst: pd.DataFrame) -> pd.DataFrame:
    """Scale-free features derived from features.features() output columns."""
    out = inst.copy()
    out["open_frac"] = out["greedy_n_open"] / out["m"]
    out["f_cv"] = out["f_std"] / out["f_mean"].where(out["f_mean"] > 0, np.nan)
    out["f_cv"] = out["f_cv"].fillna(0.0)
    out["c_cv"] = out["c_std"] / out["c_mean"]
    return out


@dataclass(frozen=True)
class Outcomes:
    """Per (instance, r) SA outcomes averaged over seeds; rows = instances, columns = grid r."""

    gap: pd.DataFrame        # mean repaired gap %, failed runs scored FAIL_GAP
    fail: pd.DataFrame       # share of runs with no solution
    feas: pd.DataFrame       # share of reads that are raw-feasible
    cpu: pd.DataFrame        # mean CPU seconds per run

    @property
    def grid(self) -> np.ndarray:
        """The r values available for every instance."""
        return np.array(sorted(self.gap.columns))


def outcomes(runs: pd.DataFrame, key: str = "instance") -> Outcomes:
    """Average the per-run CSV rows into per (instance, r) tables."""
    runs = runs.assign(
        eval_gap=runs["gap_pct"].fillna(FAIL_GAP),
        failed=runs["repaired_cost"].isna().astype(float),
        feas=runs["n_feasible_reads"] / runs["reads"],
    )
    g = runs.groupby([key, "r"])
    tab = lambda col: g[col].mean().unstack("r")  # noqa: E731
    return Outcomes(tab("eval_gap"), tab("failed"), tab("feas"), tab("cpu_time_s"))


def oracle_r(out: Outcomes) -> pd.Series:
    """Label r* per instance: lowest mean gap; ties broken by lower CPU time, then smaller r."""
    best = {}
    for inst in out.gap.index:
        cand = pd.DataFrame({"gap": out.gap.loc[inst], "cpu": out.cpu.loc[inst]})
        cand = cand.reset_index().sort_values(["gap", "cpu", "r"])
        best[inst] = float(cand["r"].iloc[0])
    return pd.Series(best, name="r_star")


def snap(r: np.ndarray | pd.Series, grid: np.ndarray) -> np.ndarray:
    """Nearest grid value to each r, measured in log space (penalties act multiplicatively)."""
    r = np.clip(np.asarray(r, dtype=float), 1e-9, None)
    idx = np.abs(np.log(r)[:, None] - np.log(grid)[None, :]).argmin(axis=1)
    return grid[idx]


def lookup(out: Outcomes, r: pd.Series) -> pd.DataFrame:
    """Outcome of using r (already on the grid) for each instance."""
    rows = []
    for inst, rv in r.items():
        rows.append({"instance": inst, "r": rv, "gap": out.gap.at[inst, rv],
                     "fail": out.fail.at[inst, rv], "feas": out.feas.at[inst, rv],
                     "cpu": out.cpu.at[inst, rv]})
    return pd.DataFrame(rows).set_index("instance")


# ---- rules: each takes training data and returns a function features -> r ----------------------

def fit_model(X: pd.DataFrame, r_star: pd.Series) -> XGBRegressor:
    """XGBoost regressor on log r*."""
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X[FEATURES], np.log(r_star.loc[X.index]))
    return model


def fit_fixed_best(out: Outcomes, train: list) -> float:
    """The single grid r with the lowest mean gap on the training instances."""
    return float(out.gap.loc[train].mean().idxmin())


def fit_bound_k(out: Outcomes, X: pd.DataFrame, train: list,
                ks: np.ndarray = np.round(np.arange(0.05, 2.0001, 0.05), 2)) -> float:
    """k for the rule r = k x (sufficient exactness bound / LP), chosen on training instances."""
    best_k, best_gap = None, np.inf
    for k in ks:
        r = pd.Series(snap(k * X.loc[train, "bound_suf_rel"], out.grid), index=train)
        g = lookup(out, r)["gap"].mean()
        if g < best_gap - 1e-12:
            best_k, best_gap = float(k), g
    return best_k


def subset(out: Outcomes, idx: list) -> Outcomes:
    """Outcomes restricted to some instances."""
    return Outcomes(out.gap.loc[idx], out.fail.loc[idx], out.feas.loc[idx], out.cpu.loc[idx])


def step_up(r: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The next grid value above each r (r itself if it is the largest)."""
    idx = np.searchsorted(grid, np.asarray(r), side="right")
    return grid[np.minimum(idx, len(grid) - 1)]


def fit_curve_model(out: Outcomes, X: pd.DataFrame, features: list[str]) -> XGBRegressor:
    """Gap-curve model: predict log(1 + gap) at penalty r from (features, log r); the rule then
    picks the r with the lowest predicted gap. Undershooting and overshooting are weighted by
    what they actually cost, instead of by distance in r as when regressing r* directly."""
    long = out.gap.stack().rename("gap").reset_index()             # instance, r, gap
    Z = X.loc[long["instance"], features].reset_index(drop=True)
    Z["log_r"] = np.log(long["r"].to_numpy())
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(Z, np.log1p(long["gap"].to_numpy()))
    return model


def predict_curve(model: XGBRegressor, X: pd.DataFrame, features: list[str], grid: np.ndarray) -> np.ndarray:
    """argmin over grid of the curve model's predicted gap, per instance."""
    preds = []
    for r in grid:
        Z = X[features].reset_index(drop=True).copy()
        Z["log_r"] = np.log(r)
        preds.append(model.predict(Z))
    return grid[np.argmin(np.vstack(preds), axis=0)]


def fit_linear(X: pd.DataFrame, r_star: pd.Series, features: list[str]):
    """Ridge regression on log r* with standardized features: unlike trees, it extrapolates."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X[features], np.log(r_star.loc[X.index]))
    return model


def predict_all(out_tr: Outcomes, X_tr: pd.DataFrame, X_te: pd.DataFrame, grid_te: np.ndarray,
                features: list[str] = FEATURES) -> dict[str, pd.Series]:
    """Every rule's r for the test instances; everything is fitted on the training data only.
    Predictions are snapped to grid_te (the penalties for which test outcomes exist)."""
    r_star = oracle_r(out_tr)
    model = XGBRegressor(**XGB_PARAMS).fit(X_tr[features], np.log(r_star.loc[X_tr.index]))
    k = fit_bound_k(out_tr, X_tr, list(X_tr.index))
    fixed = fit_fixed_best(out_tr, list(X_tr.index))
    curve = fit_curve_model(out_tr, X_tr, features)
    linear = fit_linear(X_tr, r_star, features)
    cand = np.array([r for r in grid_te if out_tr.grid.min() <= r <= out_tr.grid.max()])
    s = lambda v: pd.Series(snap(v, grid_te), index=X_te.index)  # noqa: E731
    base = snap(np.exp(model.predict(X_te[features])), grid_te)
    return {
        "model (XGBoost)": pd.Series(base, index=X_te.index),
        "model, rounded up one step": pd.Series(step_up(base, grid_te), index=X_te.index),
        "gap-curve model": pd.Series(predict_curve(curve, X_te, features, cand), index=X_te.index),
        "linear model (ridge)": s(np.exp(linear.predict(X_te[features]))),
        "best fixed r (train)": s(np.full(len(X_te), fixed)),
        f"k x exactness bound (k={k:g})": s(k * X_te["bound_suf_rel"]),
        "LP/n (pilot rule)": s(1.0 / X_te["n"]),
        "50% of LP (paper)": s(np.full(len(X_te), 0.5)),
        "100% of LP (tutorial)": s(np.full(len(X_te), 1.0)),
    }


def predict_rules(out: Outcomes, X: pd.DataFrame, train: list, test: list,
                  features: list[str] = FEATURES) -> dict[str, pd.Series]:
    """Every rule's r for the test instances, fitted only on the training instances, plus the oracle."""
    rules = predict_all(subset(out, train), X.loc[train], X.loc[test], out.grid, features)
    rules["oracle r*"] = pd.Series(oracle_r(subset(out, test)).loc[test].to_numpy(), index=test)
    return rules


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Paired bootstrap 95% CI of the mean of per-instance differences (resample instances)."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(out: Outcomes, rules: dict[str, pd.Series], ref: str = "model (XGBoost)") -> pd.DataFrame:
    """Mean gap, failures, feasibility, CPU and paired CI vs the reference rule, one row per rule."""
    ref_gap = lookup(out, rules[ref])["gap"]
    rows = []
    for name, r in rules.items():
        res = lookup(out, r)
        d = res["gap"] - ref_gap.loc[res.index]
        lo, hi = bootstrap_ci(d.to_numpy())
        rows.append({"rule": name, "mean gap %": res["gap"].mean(), "median r": float(np.median(r)),
                     "failed runs %": 100 * res["fail"].mean(), "raw-feasible reads %": 100 * res["feas"].mean(),
                     "CPU s": res["cpu"].mean(), f"gap - {ref} (95% CI)": f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]"})
    return pd.DataFrame(rows).set_index("rule")
