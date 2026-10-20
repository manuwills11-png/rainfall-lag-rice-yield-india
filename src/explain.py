"""Explainability utilities: TreeSHAP (XGBoost native), permutation importance, rainfall-response profiles.

TreeSHAP values come from ``Booster.predict(pred_contribs=True)`` which implements the exact path-dependent
TreeSHAP algorithm (Lundberg et al., 2020), so no extra dependency is required.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from src.experiment import TARGET
from src.weather_features import STAGES


def xgb_shap(model: xgb.XGBRegressor, X: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """Per-observation SHAP matrix (N x F) for an XGBoost regressor; additive to prediction with the bias column."""
    contrib = model.get_booster().predict(xgb.DMatrix(X, feature_names=list(names)), pred_contribs=True)
    out = pd.DataFrame(contrib[:, :-1], columns=list(names))
    out["_bias"] = contrib[:, -1]
    return out


def lag_profile(shap_df: pd.DataFrame, n_lags: int = 16, prefix: str = "Rain_Lag_") -> pd.DataFrame:
    """Mean |SHAP| for each pre-harvest week; normalised to a response profile summing to 1 over the 16 weeks."""
    rows = []
    for k in range(n_lags, 0, -1):
        col = f"{prefix}{k}"
        rows.append(dict(weeks_before_harvest=-k, lag=col, mean_abs_shap=float(shap_df[col].abs().mean()),
                         mean_shap=float(shap_df[col].mean())))
    df = pd.DataFrame(rows)
    df["share"] = df["mean_abs_shap"] / df["mean_abs_shap"].sum()
    return df


def stage_of_lag(k: int) -> str:
    for s, ks in STAGES.items():
        if k in ks:
            return s
    raise KeyError(k)


def stage_profile(profile: pd.DataFrame) -> pd.DataFrame:
    p = profile.copy()
    p["stage"] = [stage_of_lag(-w) for w in p["weeks_before_harvest"]]
    s = p.groupby("stage", sort=False).agg(mean_abs_shap=("mean_abs_shap", "sum"), n_weeks=("lag", "size")).reset_index()
    s["share"] = s["mean_abs_shap"] / s["mean_abs_shap"].sum()
    s["share_per_week"] = s["share"] / s["n_weeks"]
    return s


def permutation_week_importance(
    model, X: np.ndarray, y: np.ndarray, names: Sequence[str], n_repeats: int = 20, seed: int = 0, n_lags: int = 16, prefix: str = "Rain_Lag_"
) -> pd.DataFrame:
    """Increase in MAE when a single weekly lag column is permuted (model-agnostic, works for RF/XGB/Ridge)."""
    rng = np.random.default_rng(seed)
    base = np.mean(np.abs(y - model.predict(X)))
    rows = []
    for k in range(n_lags, 0, -1):
        j = list(names).index(f"{prefix}{k}")
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            inc.append(np.mean(np.abs(y - model.predict(Xp))) - base)
        rows.append(dict(weeks_before_harvest=-k, mae_increase=float(np.mean(inc)), sd=float(np.std(inc))))
    return pd.DataFrame(rows)


def group_permutation_importance(
    model, X: np.ndarray, y: np.ndarray, names: Sequence[str], groups: Dict[str, List[str]], n_repeats: int = 20, seed: int = 0
) -> pd.DataFrame:
    """Permute whole column groups jointly (e.g. all 16 lags together; the state one-hot block)."""
    rng = np.random.default_rng(seed)
    base = np.mean(np.abs(y - model.predict(X)))
    rows = []
    for g, cols in groups.items():
        idx = [list(names).index(c) for c in cols]
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, idx] = Xp[perm][:, idx]       # row-wise joint permutation keeps within-group structure
            inc.append(np.mean(np.abs(y - model.predict(Xp))) - base)
        rows.append(dict(group=g, mae_increase=float(np.mean(inc)), sd=float(np.std(inc))))
    return pd.DataFrame(rows).sort_values("mae_increase", ascending=False)
