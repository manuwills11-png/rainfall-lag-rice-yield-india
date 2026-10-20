"""Tests for weekly weather aggregation and leakage-safe lag construction."""
import numpy as np
import pandas as pd
import pytest

from src.crop_calendar import get_crop_reference_week
from src.weather_features import (N_LAGS, RAIN_LAGS, STAGES, build_lag_table, daily_to_weekly, season_summary)


def _weekly(geo="PUNJAB_1", years=(2000, 2001)):
    rows = []
    for y in years:
        for w in range(1, 54):
            rows.append(dict(stable_geographic_id=geo, iso_year=y, iso_week=w, rain_mm=float(w) + 1000 * (y - 2000), tmax_c=30.0 + w / 100, n_days=7))
    return pd.DataFrame(rows)


def _clean(state="Punjab", geo="PUNJAB_1", years=(2000, 2001)):
    return pd.DataFrame(dict(stable_geographic_id=geo, Year=list(years), State_Clean=state, **{"RICE.YIELD..Kg.per.ha.": 3000.0}))


def test_daily_to_weekly_sums_and_counts():
    dates = pd.date_range("2000-01-03", periods=14)               # Monday start -> exactly two ISO weeks
    daily = pd.DataFrame(dict(stable_geographic_id="X", date=dates, rain_mm=1.0, tmax_c=30.0))
    wk = daily_to_weekly(daily)
    assert len(wk) == 2 and (wk["rain_mm"] == 7.0).all() and (wk["n_days"] == 7).all()


def test_lag_columns_map_to_weeks_before_harvest():
    t_ref = get_crop_reference_week("Punjab")                      # 42
    out = build_lag_table(_clean(), _weekly())
    first = out.iloc[0]
    assert first["T_ref"] == t_ref
    assert first["Rain_Lag_1"] == pytest.approx(t_ref - 1)         # week 41
    assert first["Rain_Lag_16"] == pytest.approx(t_ref - 16)       # week 26
    second = out.iloc[1]                                           # year offset +1000 mm
    assert second["Rain_Lag_1"] == pytest.approx(1000 + t_ref - 1)


def test_no_lag_uses_harvest_week_or_later():
    """Zero look-ahead: changing rainfall in/after T_ref must not change any lag feature."""
    w = _weekly()
    base = build_lag_table(_clean(), w)
    t_ref = get_crop_reference_week("Punjab")
    w2 = w.copy()
    w2.loc[w2["iso_week"] >= t_ref, "rain_mm"] = 99999.0
    pert = build_lag_table(_clean(), w2)
    pd.testing.assert_frame_equal(base[RAIN_LAGS], pert[RAIN_LAGS])


def test_anchor_shift_moves_window_earlier_only():
    t_ref = get_crop_reference_week("Punjab")
    out = build_lag_table(_clean(), _weekly(), anchor_shift=8)
    assert out.iloc[0]["Rain_Lag_1"] == pytest.approx(t_ref - 8 - 1)


def test_missing_weather_gives_nan_not_zero():
    out = build_lag_table(_clean(years=(2000, 2005)), _weekly(years=(2000,)))
    assert out.loc[out["Year"] == 2005, RAIN_LAGS].isna().all().all()


def test_season_summary_consistency():
    out = build_lag_table(_clean(), _weekly())
    s = season_summary(out)
    r = out[RAIN_LAGS].to_numpy()
    assert np.allclose(s["Rain_Total16"], r.sum(1))
    assert np.allclose(s["Rain_Cum16"], s["Rain_Total16"])
    assert np.allclose(s["Rain_Last4"], out[["Rain_Lag_1", "Rain_Lag_2", "Rain_Lag_3", "Rain_Lag_4"]].sum(1))
    stage_total = s[[f"Rain_{k}" for k in STAGES]].sum(1)
    assert np.allclose(stage_total, s["Rain_Total16"])             # the 5 stages partition the 16 weeks
