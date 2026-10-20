"""Tests for the experiment engine: causality of derived features, controls and statistics."""
import numpy as np
import pandas as pd
import pytest

from src.experiment import (TARGET, add_context_features, add_rain_anomalies, paired_cluster_bootstrap,
                            shuffle_within_row, split_frame)
from src.weather_features import RAIN_LAGS


def _panel(n_dist=6, years=range(1990, 2018), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dist):
        for y in years:
            r = dict(stable_geographic_id=f"S_{d}", State_Clean="Kerala" if d % 2 else "Punjab", Year=y,
                     **{TARGET: 1000 + 10 * (y - 1990) + 50 * d + rng.normal(0, 20)})
            r.update({c: rng.gamma(2, 20) for c in RAIN_LAGS})
            rows.append(r)
    return pd.DataFrame(rows)


def test_split_is_strictly_chronological():
    p = split_frame(_panel())
    assert p["train"].Year.max() < p["validation"].Year.min() and p["validation"].Year.max() < p["test"].Year.min()


def test_context_features_only_use_the_past():
    d = _panel()
    base = add_context_features(d).set_index(["stable_geographic_id", "Year"])
    d2 = d.copy()
    d2.loc[(d2.Year >= 2010), TARGET] += 5000                     # corrupt the future
    pert = add_context_features(d2).set_index(["stable_geographic_id", "Year"])
    early = base.index.get_level_values("Year") <= 2010           # 2010 lag features depend on <=2009 only
    cols = ["Yield_Prev1", "Yield_Prev3Mean"]
    pd.testing.assert_frame_equal(base.loc[early, cols], pert.loc[early, cols])


def test_prev1_is_last_years_yield():
    d = add_context_features(_panel())
    x = d[d.stable_geographic_id == "S_0"].sort_values("Year")
    assert x["Yield_Prev1"].iloc[5] == pytest.approx(x[TARGET].iloc[4])


def test_rain_anomaly_climatology_uses_training_years_only():
    d = _panel()
    a = add_rain_anomalies(d, train_end=2009)
    d2 = d.copy()
    d2.loc[d2.Year > 2009, RAIN_LAGS] *= 10                       # change only validation/test weather
    b = add_rain_anomalies(d2, train_end=2009)
    tr = d.Year <= 2009
    pd.testing.assert_frame_equal(a.loc[tr, ["Rain_Anom_Lag_1"]], b.loc[tr, ["Rain_Anom_Lag_1"]])
    # climatology fixed => post-2009 anomaly shifts exactly with the raw change
    delta = (b.loc[~tr, "Rain_Anom_Lag_1"] - a.loc[~tr, "Rain_Anom_Lag_1"]).to_numpy()
    assert np.allclose(delta, 9 * d.loc[~tr, "Rain_Lag_1"].to_numpy())


def test_shuffle_within_row_preserves_aggregates_but_changes_order():
    d = _panel()
    s = shuffle_within_row(d, RAIN_LAGS, seed=1)
    assert np.allclose(np.sort(d[RAIN_LAGS].to_numpy(), 1), np.sort(s[RAIN_LAGS].to_numpy(), 1))
    assert np.allclose(d[RAIN_LAGS].sum(1), s[RAIN_LAGS].sum(1))
    assert not np.allclose(d[RAIN_LAGS].to_numpy(), s[RAIN_LAGS].to_numpy())


def test_paired_bootstrap_detects_a_real_difference_and_a_null():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(200), 4)
    a = np.abs(rng.normal(100, 20, 800))
    res = paired_cluster_bootstrap(a - 10, a, cl, n_boot=500)
    assert res["delta_mae"] == pytest.approx(-10) and res["ci_hi"] < 0 and res["p_boot"] < 0.05
    null = paired_cluster_bootstrap(a, a + rng.normal(0, 1, 800), cl, n_boot=500)
    assert null["ci_lo"] < 0 < null["ci_hi"]
