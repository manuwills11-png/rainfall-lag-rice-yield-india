"""Panel fixed-effects distributed-lag regression: the *explanatory* counterpart of the predictive experiments.

log(yield_it) = sum_k beta_k * RainAnomaly_{it,k} + district FE + year FE + e_it     (k = 1..16 weeks before harvest)

Rain anomalies are in units of 100 mm so that beta_k is the proportional yield change per +100 mm in week -k
(multiply by 100 for %). Standard errors are clustered by district. Because the analysis is explanatory, it uses all
years (1990-2017); it is never used for model selection in the predictive experiments.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.experiment import ANOM_LAGS, TARGET
from src.weather_features import STAGES


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ly"] = np.log(d[TARGET])
    d["dist"] = d["stable_geographic_id"]
    for c in ANOM_LAGS:
        d[c] = d[c] / 100.0
    return d


def fit_fe(df: pd.DataFrame, regressors: Sequence[str], fe: str = "C(dist) + C(Year)"):
    d = _prep(df) if "ly" not in df.columns else df
    f = "ly ~ " + " + ".join(regressors) + " + " + fe
    return smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["dist"])[0]})


def lag_coefficients(df: pd.DataFrame, fe: str = "C(dist) + C(Year)") -> Dict[str, object]:
    """Distributed-lag coefficients (percent per +100 mm) with 95% CI and two joint Wald tests."""
    d = _prep(df)
    m = fit_fe(d, ANOM_LAGS, fe)
    ci = m.conf_int().loc[ANOM_LAGS]
    tab = pd.DataFrame({
        "weeks_before_harvest": [-int(c.split("_")[-1]) for c in ANOM_LAGS],
        "coef_pct_per_100mm": m.params[ANOM_LAGS].to_numpy() * 100,
        "ci_lo": ci[0].to_numpy() * 100, "ci_hi": ci[1].to_numpy() * 100,
        "p_value": m.pvalues[ANOM_LAGS].to_numpy(),
    })
    names = list(m.params.index)
    # H0a: all 16 lag coefficients are zero (weather has no timing-resolved effect)
    Ra = np.zeros((16, len(names)))
    for i, c in enumerate(ANOM_LAGS):
        Ra[i, names.index(c)] = 1
    wa = m.wald_test(Ra, scalar=True)
    # H0b: all 16 coefficients are equal (timing does not matter; only the seasonal total does)
    Rb = np.zeros((15, len(names)))
    for i in range(15):
        Rb[i, names.index(ANOM_LAGS[i])] = 1; Rb[i, names.index(ANOM_LAGS[i + 1])] = -1
    wb = m.wald_test(Rb, scalar=True)
    return dict(table=tab, model=m, n=int(m.nobs), r2=float(m.rsquared),
                wald_zero=(float(wa.statistic), float(wa.pvalue)), wald_equal=(float(wb.statistic), float(wb.pvalue)))


def stage_coefficients(df: pd.DataFrame, fe: str = "C(dist) + C(Year)") -> pd.DataFrame:
    d = _prep(df)
    cols = []
    for s, ks in STAGES.items():
        c = f"Anom_{s}"
        d[c] = d[[f"Rain_Anom_Lag_{k}" for k in ks]].sum(axis=1)
        cols.append(c)
    m = fit_fe(d, cols, fe)
    ci = m.conf_int().loc[cols]
    return pd.DataFrame({"stage": list(STAGES), "coef_pct_per_100mm": m.params[cols].to_numpy() * 100,
                         "ci_lo": ci[0].to_numpy() * 100, "ci_hi": ci[1].to_numpy() * 100, "p_value": m.pvalues[cols].to_numpy()})
