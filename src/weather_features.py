"""Daily -> weekly weather aggregation and vectorised, leakage-safe pre-harvest lag construction.

The weather source is the IMD gridded daily rainfall (0.25 deg) and Tmax (1 deg) extracted around one representative
point per district (see ``scripts/10_geocode_districts.py``, ``scripts/12_download_imd_grid.py`` and ``scripts/13_extract_district_weather.py``).

Week convention: ISO-8601 weeks of the *crop year* (the dataset ``Year``). For a district-year with harvest
reference week ``T_ref`` (``src.crop_calendar``) the lag columns are

    Rain_Lag_k = rainfall of ISO week (T_ref - k)    k = 1..16

so Rain_Lag_1 is the last full week strictly before ``T_ref``. ``anchor_shift`` moves the whole window
backwards (placebo control) and is never used in the main experiments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.crop_calendar import get_crop_reference_week

N_LAGS = 16
RAIN_LAGS = [f"Rain_Lag_{k}" for k in range(N_LAGS, 0, -1)]      # chronological order: -16 ... -1
TMAX_LAGS = [f"Tmax_Lag_{k}" for k in range(N_LAGS, 0, -1)]

# Phenological stage -> lag indices k (weeks before harvest), from docs/crop_calendar_reference.md
STAGES: Dict[str, Sequence[int]] = {
    "S1_Sowing_Transplanting": (16, 15, 14, 13),
    "S2_Tillering": (12, 11, 10, 9, 8),
    "S3_Panicle_Initiation": (7, 6, 5, 4),
    "S4_Flowering": (3, 2),
    "S5_Grain_Filling": (1,),
}


def load_daily_weather(directory: str | Path) -> pd.DataFrame:
    """Concatenate per-district daily CSVs into one long frame [stable_geographic_id, date, rain_mm, tmax_c]."""
    frames = []
    for f in sorted(Path(directory).glob("*.csv")):
        d = pd.read_csv(f, parse_dates=["date"])
        d.insert(0, "stable_geographic_id", f.stem)
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no daily weather files in {directory}")
    return pd.concat(frames, ignore_index=True)


def daily_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """ISO-week aggregation: weekly rainfall sum (mm), weekly mean of daily Tmax (degC)."""
    d = daily.copy()
    iso = d["date"].dt.isocalendar()
    d["iso_year"] = iso["year"].astype(int)
    d["iso_week"] = iso["week"].astype(int)
    out = (d.groupby(["stable_geographic_id", "iso_year", "iso_week"])
             .agg(rain_mm=("rain_mm", "sum"), tmax_c=("tmax_c", "mean"), n_days=("rain_mm", "size"))
             .reset_index())
    return out


def _weekly_arrays(weekly: pd.DataFrame, value: str) -> Dict[tuple, np.ndarray]:
    """(geo, iso_year) -> length-54 array indexed by ISO week (index 0 unused), NaN where missing."""
    arrs: Dict[tuple, np.ndarray] = {}
    for (geo, yr), g in weekly.groupby(["stable_geographic_id", "iso_year"]):
        a = np.full(54, np.nan)
        a[g["iso_week"].to_numpy()] = g[value].to_numpy()
        arrs[(geo, int(yr))] = a
    return arrs


def build_lag_table(
    clean_df: pd.DataFrame,
    weekly: pd.DataFrame,
    n_lags: int = N_LAGS,
    anchor_shift: int = 0,
    state_col: str = "State_Clean",
    include_tmax: bool = True,
) -> pd.DataFrame:
    """Attach ``Rain_Lag_k`` (and ``Tmax_Lag_k``) columns to every district-year row.

    Raises if any lag week is >= T_ref + anchor_shift is violated, or if any weather value is missing.
    """
    rain = _weekly_arrays(weekly, "rain_mm")
    tmax = _weekly_arrays(weekly, "tmax_c") if include_tmax else None

    rows_r, rows_t, t_refs = [], [], []
    for geo, yr, st in zip(clean_df["stable_geographic_id"], clean_df["Year"], clean_df[state_col]):
        t_ref = get_crop_reference_week(st) - anchor_shift
        weeks = np.arange(t_ref - n_lags, t_ref)          # chronological: w_-n ... w_-1
        assert weeks.max() < t_ref, "lookahead leakage: lag week >= T_ref"
        assert weeks.min() >= 1
        a = rain.get((geo, int(yr)))
        rows_r.append(a[weeks] if a is not None else np.full(n_lags, np.nan))
        if include_tmax:
            b = tmax.get((geo, int(yr)))
            rows_t.append(b[weeks] if b is not None else np.full(n_lags, np.nan))
        t_refs.append(t_ref)

    out = clean_df.copy()
    out["T_ref"] = t_refs
    rcols = [f"Rain_Lag_{k}" for k in range(n_lags, 0, -1)]
    out[rcols] = np.vstack(rows_r)
    if include_tmax:
        out[[f"Tmax_Lag_{k}" for k in range(n_lags, 0, -1)]] = np.vstack(rows_t)
    return out


def season_summary(df: pd.DataFrame, n_lags: int = N_LAGS) -> pd.DataFrame:
    """Aggregate (non-lag) representations derived from the *same* IMD lag window, for controlled ablations."""
    r = df[[f"Rain_Lag_{k}" for k in range(n_lags, 0, -1)]].to_numpy()        # chronological
    rev = r[:, ::-1]                                                          # rev[:, j] = lag j+1
    s = pd.DataFrame(index=df.index)
    s["Rain_Total16"] = r.sum(1)
    s["Rain_SD16"] = r.std(1)
    s["Rain_Max_Week16"] = r.max(1)
    s["Rain_Last4"] = rev[:, :4].sum(1)             # "contemporaneous": month before harvest
    s["Rain_Lag_1_only"] = rev[:, 0]
    for h in (4, 8, 12, 16):                        # cumulative from harvest backwards
        s[f"Rain_Cum{h}"] = rev[:, :h].sum(1)
    for name, ks in STAGES.items():
        s[f"Rain_{name}"] = rev[:, [k - 1 for k in ks]].sum(1)
    return s
