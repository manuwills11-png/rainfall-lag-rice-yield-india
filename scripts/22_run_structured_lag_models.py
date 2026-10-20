"""Pre-specified low-dimensional ("structured") lag models - extension experiment E9.

Hypothesis fixed BEFORE running: the 16 free weekly lags are statistically wasteful; a parsimonious structure suggested by the
panel analysis (smooth lag profile, phenological stages, asymmetric drought response, a train-only learned weather index)
may recover forecasting skill. Candidates are listed in CANDIDATES (8 incl. the F8 reference) and never changed after seeing test results.
Selection rule: per model, the candidate with the lowest VALIDATION MAE. Test years are scored once. Multiplicity: Holm-adjusted p-values
over the 7 new candidates x 3 models vs the F8 total-anomaly reference.
Output: results/tables/p2_structured_lag_{metrics,selection,bootstrap}.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from numpy.polynomial import legendre
from src.experiment import *                                 # noqa
from src.panel import lag_coefficients
from src.weather_features import STAGES

SEEDS = (42, 7, 123, 2024, 99)
D = add_rain_anomalies(pd.read_csv(ROOT / "data/processed/phase2_lag_features.csv.gz"), 2009)
A = D[ANOM_LAGS].to_numpy() / 100.0                          # chronological: week -16 ... -1  (100 mm units)
x = np.linspace(-1, 1, 16)

# smooth distributed lag: Legendre basis of degree d over the 16 weeks
for d in (2, 3):
    for j in range(d + 1):
        D[f"Sm{d}_{j}"] = A @ legendre.legval(x, np.eye(d + 1)[j])
# stage sums of anomalies
for s, ks in STAGES.items():
    D[f"StAn_{s}"] = D[[f"Rain_Anom_Lag_{k}" for k in ks]].sum(axis=1) / 100.0
# asymmetric total
tot = D["Rain_Anom_Total16"] / 100.0
D["Tot_neg"], D["Tot_pos"] = np.minimum(tot, 0), np.maximum(tot, 0)
# train-only learned weather index (panel FE coefficients estimated on 1990-2009 only, then smoothed over weeks)
beta = lag_coefficients(D[D.Year <= 2009])["table"].sort_values("weeks_before_harvest").coef_pct_per_100mm.to_numpy() / 100
beta_s = np.convolve(np.pad(beta, 1, mode="edge"), np.ones(3) / 3, mode="valid")
D["PanelIdx"] = A @ beta_s
D["PIdx_neg"], D["PIdx_pos"] = np.minimum(D.PanelIdx, 0), np.maximum(D.PanelIdx, 0)
ST = [f"StAn_{s}" for s in STAGES]

CANDIDATES = {
    "F8_Total_Anomaly (reference)": ["Rain_Anom_Total16", "IMD_Tmax"],
    "S1_Smooth2": [f"Sm2_{j}" for j in range(3)] + ["IMD_Tmax"],
    "S2_Smooth3": [f"Sm3_{j}" for j in range(4)] + ["IMD_Tmax"],
    "S3_StageAnomaly": ST + ["IMD_Tmax"],
    "S4_AsymTotal": ["Tot_neg", "Tot_pos", "IMD_Tmax"],
    "S5_PanelIndex": ["PanelIdx", "IMD_Tmax"],
    "S6_PanelIndexAsym": ["PIdx_neg", "PIdx_pos", "IMD_Tmax"],
    "S7_StageAnomaly+Asym": ST + ["Tot_neg", "Tot_pos", "IMD_Tmax"],
}
mets, preds = [], []
for name, cols in CANDIDATES.items():
    p, m, _, _ = run_tabular(D, name, cols, seeds=SEEDS, target_mode="anomaly")
    mets.append(m); preds.append(p[p.split == "test"]); print("done", name, flush=True)
M = pd.concat(mets); P = pd.concat(preds)
M.to_csv(ROOT / "results/tables/p2_structured_lag_metrics.csv", index=False)

# selection on validation only
S = M[M.split == "validation"].groupby(["feature_set", "model"]).MAE.mean().reset_index()
new = S[~S.feature_set.str.contains("reference")]
sel = new.loc[new.groupby("model").MAE.idxmin()].rename(columns={"MAE": "val_MAE"})
ref_val = S[S.feature_set.str.contains("reference")].set_index("model").MAE
sel["ref_val_MAE"] = sel.model.map(ref_val)
T = M[M.split == "test"].groupby(["feature_set", "model"]).MAE.mean().reset_index().rename(columns={"MAE": "test_MAE"})
sel = sel.merge(T, on=["feature_set", "model"])
sel.to_csv(ROOT / "results/tables/p2_structured_lag_selection.csv", index=False)

# paired bootstrap vs reference, Holm over all 21 comparisons
def ens(fs, mod):
    x_ = P[(P.feature_set == fs) & (P.model == mod)].groupby(["stable_geographic_id", "Year"]).agg(y=("Actual", "first"), p=("Predicted", "mean"))
    x_["ae"] = (x_.y - x_.p).abs(); return x_
rows = []
for mod in ("Ridge", "RF", "XGB"):
    r = ens("F8_Total_Anomaly (reference)", mod)
    for fs in CANDIDATES:
        if "reference" in fs: continue
        e = ens(fs, mod); idx = e.index.intersection(r.index)
        rows.append(dict(model=mod, candidate=fs, **paired_cluster_bootstrap(e.loc[idx, "ae"].values, r.loc[idx, "ae"].values, np.array([i[0] for i in idx]))))
B = pd.DataFrame(rows).sort_values("p_boot")
m_ = len(B); B["p_holm"] = np.minimum(1, np.maximum.accumulate((m_ - np.arange(m_)) * B.p_boot.values))
B.to_csv(ROOT / "results/tables/p2_structured_lag_bootstrap.csv", index=False)
print(sel.round(2).to_string(index=False)); print(B.round(3).to_string(index=False))
print(M[M.split == "test"].groupby(["feature_set", "model"]).MAE.mean().unstack().round(1))
