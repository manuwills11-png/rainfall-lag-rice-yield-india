"""Build the Phase-2 modelling table: yield + aggregates + 16 weekly rainfall lags (+Tmax lags) + context features.

Output: data/processed/phase2_lag_features.csv.gz and results/tables/p2_lag_dataset_audit.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.weather_features import build_lag_table, season_summary, RAIN_LAGS, TMAX_LAGS
from src.experiment import add_context_features

clean = pd.read_csv(ROOT / "data/processed/phase1_clean.csv")
weekly = pd.read_csv(ROOT / "data/external/district_weekly_weather.csv.gz")
lag = build_lag_table(clean, weekly)
assert lag[RAIN_LAGS + TMAX_LAGS].notna().all().all(), "missing weather in lag window"
assert (lag[RAIN_LAGS] >= 0).all().all()
lag = pd.concat([lag, season_summary(lag)], axis=1)
lag = add_context_features(lag)
coords = pd.read_csv(ROOT / "data/external/district_coordinates_final.csv")[["stable_geographic_id", "lat", "lon", "source", "r_train_final"]]
lag = lag.merge(coords, on="stable_geographic_id", how="left")
(ROOT / "data/processed").mkdir(exist_ok=True)
lag.to_csv(ROOT / "data/processed/phase2_lag_features.csv.gz", index=False)

audit = pd.DataFrame([
    ("rows", len(lag)), ("districts", lag.stable_geographic_id.nunique()), ("lag columns", len(RAIN_LAGS)),
    ("missing rain lags", int(lag[RAIN_LAGS].isna().sum().sum())), ("missing tmax lags", int(lag[TMAX_LAGS].isna().sum().sum())),
    ("min T_ref", int(lag.T_ref.min())), ("max T_ref", int(lag.T_ref.max())),
    ("max lag week index (must be < T_ref)", int((lag.T_ref - 1).max())),
    ("mean 16-week rainfall (mm)", round(lag.Rain_Total16.mean(), 1)),
    ("share of observations with zero rain in some week", round((lag[RAIN_LAGS].eq(0).any(axis=1)).mean(), 3)),
], columns=["check", "value"])
(ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
audit.to_csv(ROOT / "results/tables/p2_lag_dataset_audit.csv", index=False)
print(audit.to_string(index=False)); print(lag.shape)
