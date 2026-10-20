"""Experiment engine for the rainfall-lag ablation study (tabular + sequence models, controls, statistics).

Everything here is leakage-aware by construction:
* encoders / scalers / climatologies are fitted on the *training* years only;
* hyper-parameters are chosen on the validation years only;
* the test years are touched once per (feature set, model, seed) to produce predictions;
* context features (previous-year yield) look strictly backwards in time.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation import calculate_metrics
from src.weather_features import RAIN_LAGS, TMAX_LAGS, STAGES

TARGET = "RICE.YIELD..Kg.per.ha."
SPLITS = {"train": (1990, 2009), "validation": (2010, 2013), "test": (2014, 2017)}
CAT = ["State_Clean"]
CTX = ["Yield_Prev1", "Yield_Prev3Mean"]

STAGE_COLS = [f"Rain_{s}" for s in STAGES]
ANOM_LAGS = [c.replace("Rain_Lag", "Rain_Anom_Lag") for c in RAIN_LAGS]
BASE_COL = "Yield_Prev3Mean"          # district baseline used by the anomaly formulation
FEATURE_SETS: Dict[str, List[str]] = {
    "F0_Aggregate_IMD":   ["IMD_RF", "IMD_Tmax", "SD_RF_jun_nov"],              # Phase-1 baseline
    "F1_Aggregate_IMDgrid":  ["Rain_Total16", "Rain_SD16", "IMD_Tmax"],            # same IMD-grid source, no timing
    "F2_Contemporaneous": ["Rain_Last4", "Rain_Lag_1_only", "IMD_Tmax"],        # month before harvest only
    "F3_Cumulative":      ["Rain_Cum4", "Rain_Cum8", "Rain_Cum12", "Rain_Cum16", "IMD_Tmax"],
    "F4_Stage":           STAGE_COLS + ["IMD_Tmax"],                            # 5 phenological stage sums
    "F5_Lag16":           RAIN_LAGS + ["IMD_Tmax"],                             # proposed: 16 weekly lags
    "F6_Lag16_PlusTmaxLags": RAIN_LAGS + TMAX_LAGS + ["IMD_Tmax"],              # extension: weekly temperature
    "F7_Lag16_Anomaly":   ANOM_LAGS + ["IMD_Tmax"],                             # extension: lags as departures from normal
    "F8_Total_Anomaly":   ["Rain_Anom_Total16", "IMD_Tmax"],                    # aggregate counterpart of F7
}


# ----------------------------------------------------------------------------------------------- data
def split_frame(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {k: df[(df["Year"] >= a) & (df["Year"] <= b)].copy() for k, (a, b) in SPLITS.items()}
    assert out["train"]["Year"].max() < out["validation"]["Year"].min() <= out["validation"]["Year"].max() < out["test"]["Year"].min()
    return out


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Previous-season district yield and mean of the previous three available seasons (backward-looking only).

    Known at forecast time because last season's statistics are published before the new crop is harvested.
    A district's first observation has no history; it is filled with the previous-year state mean, then the
    previous-year national mean, and - only for the very first survey year (1990, a training year) - with that
    year's own national mean. No later year ever influences an earlier row.
    """
    d = df.sort_values(["stable_geographic_id", "Year"]).copy()
    g = d.groupby("stable_geographic_id")[TARGET]
    d["Yield_Prev1"] = g.shift(1)
    d["Yield_Prev3Mean"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    st = d.groupby(["State_Clean", "Year"])[TARGET].mean().rename("_sm").reset_index()
    st["Year"] += 1                                                       # previous-year state mean
    nat = d.groupby("Year")[TARGET].mean()
    d = d.merge(st, on=["State_Clean", "Year"], how="left")
    nat_prev = (d["Year"] - 1).map(nat)
    nat_same = d["Year"].map(nat)
    fill = d["_sm"].fillna(nat_prev).fillna(nat_same.where(d["Year"] == d["Year"].min()))
    for c in CTX:
        d[c] = d[c].fillna(fill)
    return d.drop(columns="_sm").sort_index().reset_index(drop=True)


def add_rain_anomalies(df: pd.DataFrame, train_end: int = 2009) -> pd.DataFrame:
    """Rain_Anom_Lag_k = Rain_Lag_k - district climatology of that lag (training years only; state fallback)."""
    from src.weather_features import RAIN_LAGS
    d = df.copy()
    tr = d[d["Year"] <= train_end]
    clim_d = tr.groupby("stable_geographic_id")[RAIN_LAGS].mean()
    clim_s = tr.groupby("State_Clean")[RAIN_LAGS].mean()
    base = clim_d.reindex(d["stable_geographic_id"]).to_numpy()
    fb = clim_s.reindex(d["State_Clean"]).to_numpy()
    base = np.where(np.isnan(base), fb, base)
    anom = d[RAIN_LAGS].to_numpy() - base
    for j, c in enumerate(RAIN_LAGS):
        d[c.replace("Rain_Lag", "Rain_Anom_Lag")] = anom[:, j]
    d["Rain_Anom_Total16"] = anom.sum(1)
    return d


def design_matrices(parts: Dict[str, pd.DataFrame], num_cols: Sequence[str], cat_cols: Sequence[str] = CAT):
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(parts["train"][list(cat_cols)])
    names = list(num_cols) + list(enc.get_feature_names_out(cat_cols))
    X = {k: np.hstack([v[list(num_cols)].to_numpy(float), enc.transform(v[list(cat_cols)])]) for k, v in parts.items()}
    y = {k: v[TARGET].to_numpy(float) for k, v in parts.items()}
    return X, y, names, enc


# --------------------------------------------------------------------------------------------- models
RF_DEPTHS = (8, 12, 16, 20)
XGB_GRID = ({"max_depth": 4, "learning_rate": 0.05}, {"max_depth": 6, "learning_rate": 0.05}, {"max_depth": 6, "learning_rate": 0.03})
RIDGE_ALPHAS = (0.1, 1, 10, 100, 1000)


def _rf(seed, **kw):
    return RandomForestRegressor(n_estimators=200, min_samples_split=5, min_samples_leaf=2, max_features="sqrt",
                                 random_state=seed, n_jobs=-1, **kw)


def _xgb(seed, **kw):
    return xgb.XGBRegressor(n_estimators=250, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                            random_state=seed, n_jobs=-1, **kw)


def _mae(a, b):
    return float(np.mean(np.abs(a - b)))


def tune(model: str, X, y, seed: int = 42) -> Dict:
    """Pick hyper-parameters by validation MAE (same grids as Phase 1 for every feature set)."""
    best, best_mae = None, np.inf
    if model == "Ridge":
        grid = [{"alpha": a} for a in RIDGE_ALPHAS]
        build = lambda p: make_pipeline(StandardScaler(), Ridge(**p))
    elif model == "RF":
        grid = [{"max_depth": d} for d in RF_DEPTHS]
        build = lambda p: _rf(seed, **p)
    elif model == "XGB":
        grid = list(XGB_GRID)
        build = lambda p: _xgb(seed, **p)
    else:
        raise ValueError(model)
    for p in grid:
        m = build(p).fit(X["train"], y["train"])
        mae = _mae(y["validation"], m.predict(X["validation"]))
        if mae < best_mae:
            best, best_mae = p, mae
    return best


def fit_model(model: str, params: Dict, X, y, seed: int):
    if model == "Ridge":
        m = make_pipeline(StandardScaler(), Ridge(**params))
    elif model == "RF":
        m = _rf(seed, **params)
    else:
        m = _xgb(seed, **params)
    return m.fit(X["train"], y["train"])


def run_tabular(
    data: pd.DataFrame,
    feature_set: str,
    num_cols: Sequence[str],
    models: Iterable[str] = ("Ridge", "RF", "XGB"),
    seeds: Sequence[int] = (42,),
    tuned: Optional[Dict[Tuple[str, str], Dict]] = None,
    keep_models: bool = False,
    target_mode: str = "level",
):
    """Train every model x seed for one feature set. Returns (predictions long frame, metrics frame, params, models).

    target_mode="level":   model predicts yield (kg/ha) directly (Phase-1 formulation).
    target_mode="anomaly": model predicts yield - Yield_Prev3Mean (district baseline known before the season);
                           predictions are converted back to kg/ha before any metric is computed.
    """
    parts = split_frame(data)
    X, y, names, enc = design_matrices(parts, num_cols)
    base = {k: (v[BASE_COL].to_numpy(float) if target_mode == "anomaly" else np.zeros(len(v))) for k, v in parts.items()}
    y_true = y
    y = {k: y_true[k] - base[k] for k in y_true}
    preds, mets, params_out, kept = [], [], {}, {}
    for model in models:
        key = (feature_set, model)
        p = tuned[key] if tuned and key in tuned else tune(model, X, y, seed=42)
        params_out[key] = p
        for seed in (seeds if model != "Ridge" else (42,)):
            m = fit_model(model, p, X, y, seed)
            if keep_models:
                kept[(feature_set, model, seed)] = (m, names)
            for split in ("train", "validation", "test"):
                yh = m.predict(X[split]) + base[split]
                mets.append(dict(feature_set=feature_set, model=model, seed=seed, split=split, **calculate_metrics(y_true[split], yh), n=len(yh)))
                if split != "train":
                    d = parts[split][["stable_geographic_id", "State_Clean", "Year"]].copy()
                    d.insert(0, "split", split); d.insert(0, "seed", seed); d.insert(0, "model", model); d.insert(0, "feature_set", feature_set)
                    d["Actual"] = y_true[split]; d["Predicted"] = yh
                    preds.append(d)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(mets), params_out, kept


# -------------------------------------------------------------------------------------- naive baselines
def naive_baselines(data: pd.DataFrame) -> pd.DataFrame:
    """State-mean, district-mean (train years) and persistence (previous-year yield) reference forecasts."""
    parts = split_frame(data)
    tr = parts["train"]
    sm = tr.groupby("State_Clean")[TARGET].mean()
    dm = tr.groupby("stable_geographic_id")[TARGET].mean()
    rows = []
    for split in ("validation", "test"):
        v = parts[split]
        cands = {
            "Naive: train state mean": v["State_Clean"].map(sm).fillna(tr[TARGET].mean()),
            "Naive: train district mean": v["stable_geographic_id"].map(dm).fillna(v["State_Clean"].map(sm)),
            "Naive: persistence (prev-year yield)": v["Yield_Prev1"],
        }
        for name, yh in cands.items():
            rows.append(dict(model=name, split=split, **calculate_metrics(v[TARGET].to_numpy(), yh.to_numpy()), n=len(v)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------- statistics
def paired_cluster_bootstrap(
    err_a: np.ndarray, err_b: np.ndarray, clusters: np.ndarray, n_boot: int = 4000, seed: int = 0
) -> Dict[str, float]:
    """Mean paired difference in absolute error (a - b) with a district-cluster bootstrap 95% CI.

    Negative delta means model *a* has lower error than *b*. Clustering by district accounts for the
    serial correlation of errors across the four test years.
    """
    rng = np.random.default_rng(seed)
    diff = err_a - err_b
    cl = pd.factorize(clusters)[0]
    k = cl.max() + 1
    s = np.bincount(cl, weights=diff, minlength=k)
    c = np.bincount(cl, minlength=k).astype(float)
    idx = rng.integers(0, k, size=(n_boot, k))
    boot = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot >= 0).mean(), (boot <= 0).mean())
    return dict(delta_mae=float(diff.mean()), ci_lo=float(lo), ci_hi=float(hi), p_boot=float(min(1.0, p)))


def mean_seed_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """Average predictions over seeds (ensemble of seeds) -> one prediction per observation."""
    keys = ["feature_set", "model", "split", "stable_geographic_id", "State_Clean", "Year", "Actual"]
    return pred.groupby(keys, as_index=False)["Predicted"].mean()


def summarise_seeds(met: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    m = met[met["split"] == split]
    g = m.groupby(["feature_set", "model"])
    out = g.agg(MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"), RMSE_mean=("RMSE", "mean"),
                R2_mean=("R2", "mean"), R2_sd=("R2", "std"), seeds=("seed", "nunique")).reset_index()
    return out


# ------------------------------------------------------------------------------------------- controls
def shuffle_within_row(df: pd.DataFrame, cols: Sequence[str], seed: int = 0) -> pd.DataFrame:
    """Randomly permute the values of ``cols`` independently inside every row.

    Preserves each observation's multiset of weekly values (hence every aggregate: total, SD, max) but destroys
    *when* the rain fell. A model that gains from timing must lose that gain on this control.
    """
    rng = np.random.default_rng(seed)
    d = df.copy()
    v = d[list(cols)].to_numpy().copy()
    for i in range(len(v)):
        v[i] = v[i][rng.permutation(v.shape[1])]
    d[list(cols)] = v
    return d


def rolling_origin_eval(
    data: pd.DataFrame,
    feature_set: str,
    num_cols: Sequence[str],
    test_years: Sequence[int],
    model: str,
    params: Dict,
    target_mode: str = "level",
    seeds: Sequence[int] = (42,),
) -> pd.DataFrame:
    """Expanding-window one-year-ahead evaluation: train on all years < T, predict year T (params fixed)."""
    rows = []
    for T in test_years:
        d = add_rain_anomalies(data, train_end=T - 1) if any(c.startswith("Rain_Anom") for c in num_cols) else data
        tr, te = d[d["Year"] < T], d[d["Year"] == T]
        parts = {"train": tr, "test": te}
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(tr[CAT])
        Xtr = np.hstack([tr[list(num_cols)].to_numpy(float), enc.transform(tr[CAT])])
        Xte = np.hstack([te[list(num_cols)].to_numpy(float), enc.transform(te[CAT])])
        btr = tr[BASE_COL].to_numpy(float) if target_mode == "anomaly" else 0.0
        bte = te[BASE_COL].to_numpy(float) if target_mode == "anomaly" else 0.0
        for seed in (seeds if model != "Ridge" else (42,)):
            m = fit_model(model, params, {"train": Xtr}, {"train": tr[TARGET].to_numpy(float) - btr}, seed)
            yh = m.predict(Xte) + bte
            for g, a, p_ in zip(te["stable_geographic_id"], te[TARGET], yh):
                rows.append(dict(feature_set=feature_set, model=model, seed=seed, test_year=T, stable_geographic_id=g, Actual=a, Predicted=p_))
    return pd.DataFrame(rows)
