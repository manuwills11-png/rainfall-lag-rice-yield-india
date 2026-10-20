"""Rolling-origin (2011-2017, expanding window) check of the structured lag models S1/S2 vs the F8 total anomaly and F5 free lags.
Hyper-parameters fixed from the main tier-C tuning of F8 (same for all candidates => no extra tuning advantage)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from numpy.polynomial import legendre
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.experiment import *                                  # noqa

HP = json.load(open(ROOT / "results/tables/p2_main_hyperparameters.json"))
D0 = pd.read_csv(ROOT / "data/processed/phase2_lag_features.csv.gz")
x = np.linspace(-1, 1, 16)
rows = []
for T in range(2011, 2018):
    D = add_rain_anomalies(D0, T - 1); A = D[ANOM_LAGS].to_numpy() / 100
    for d in (2, 3):
        for j in range(d + 1):
            D[f"Sm{d}_{j}"] = A @ legendre.legval(x, np.eye(d + 1)[j])
    sets = {"F8_Total_Anomaly": ["Rain_Anom_Total16", "IMD_Tmax"], "F5_Lag16": RAIN_LAGS_ if (RAIN_LAGS_ := [f"Rain_Lag_{k}" for k in range(16, 0, -1)]) else None,
            "S1_Smooth2": [f"Sm2_{j}" for j in range(3)] + ["IMD_Tmax"], "S2_Smooth3": [f"Sm3_{j}" for j in range(4)] + ["IMD_Tmax"]}
    sets["F5_Lag16"] = sets["F5_Lag16"] + ["IMD_Tmax"]
    tr, te = D[D.Year < T], D[D.Year == T]
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(tr[CAT])
    for fs, cols in sets.items():
        Xtr = np.hstack([tr[cols].to_numpy(float), enc.transform(tr[CAT])]); Xte = np.hstack([te[cols].to_numpy(float), enc.transform(te[CAT])])
        for mod in ("RF", "XGB"):
            for seed in (42, 7, 123):
                m = fit_model(mod, HP[f"C_anomaly|F8_Total_Anomaly|{mod}"], {"train": Xtr}, {"train": tr[TARGET].to_numpy() - tr[BASE_COL].to_numpy()}, seed)
                yh = m.predict(Xte) + te[BASE_COL].to_numpy()
                rows.append(pd.DataFrame(dict(feature_set=fs, model=mod, seed=seed, test_year=T, stable_geographic_id=te.stable_geographic_id.values, Actual=te[TARGET].values, Predicted=yh)))
    print("fold", T, flush=True)
R = pd.concat(rows); R.to_csv(ROOT / "results/predictions/p2_structured_rolling_origin.csv.gz", index=False)
E=R.groupby(["feature_set","model","test_year","stable_geographic_id"],as_index=False).agg(Actual=("Actual","first"),Predicted=("Predicted","mean"))
E["ae"]=(E.Actual-E.Predicted).abs()
print(E.groupby(["model","feature_set"]).ae.mean().unstack().round(2))
out=[]
for mod in ("RF","XGB"):
    ref=E[(E.model==mod)&(E.feature_set=="F8_Total_Anomaly")].set_index(["test_year","stable_geographic_id"]).ae
    for fs in ("S1_Smooth2","S2_Smooth3","F5_Lag16"):
        e=E[(E.model==mod)&(E.feature_set==fs)].set_index(["test_year","stable_geographic_id"]).ae; idx=e.index.intersection(ref.index)
        out.append(dict(model=mod,candidate=fs,**paired_cluster_bootstrap(e.loc[idx].values,ref.loc[idx].values,np.array([i[1] for i in idx]))))
B=pd.DataFrame(out); B.to_csv(ROOT/"results/tables/p2_structured_rolling_bootstrap.csv",index=False); print(B.round(3).to_string(index=False))
pd.concat([E.groupby(["model","feature_set","test_year"]).ae.mean().reset_index()]).to_csv(ROOT/"results/tables/p2_structured_rolling_by_year.csv",index=False)
