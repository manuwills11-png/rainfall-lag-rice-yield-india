"""Build + execute notebook 12: pre-specified structured (low-dimensional) lag models."""
from nb_common import build, code, md

cells = [
md(r"""
# 12 · Structured lag models: can a parsimonious lag structure recover forecasting skill?

Notebooks 07-09 showed that 16 *free* weekly lags are statistically wasteful. The panel analysis (notebook 10) suggested structure: a smooth profile over weeks, phenological stages and an asymmetric (drought-dominated) response.
`scripts/22_run_structured_lag_models.py` therefore tested **eight pre-specified candidates** (never modified after seeing results), all in the anomaly formulation (tier C):

| Code | Representation (+ state FE, `IMD_Tmax`) | Parameters |
|---|---|---|
| F8 (ref) | seasonal total anomaly | 1 |
| S1 / S2 | smooth distributed lag: Legendre basis of degree 2 / 3 over the 16 weeks | 3 / 4 |
| S3 | 5 stage-sum anomalies | 5 |
| S4 | total anomaly split into deficit and surplus | 2 |
| S5 / S6 | train-only learned panel-FE weather index (and its deficit/surplus split) | 1 / 2 |
| S7 | stage sums + deficit/surplus | 7 |

Rules fixed beforehand: selection per model on **validation MAE only**; test years scored once; Holm correction over the 21 (7 candidates × 3 models) paired comparisons against F8.
"""),
code(r"""
from src.experiment import *
M = pd.read_csv(TAB/"p2_structured_lag_metrics.csv"); SEL = pd.read_csv(TAB/"p2_structured_lag_selection.csv"); B = pd.read_csv(TAB/"p2_structured_lag_bootstrap.csv")
RB = pd.read_csv(TAB/"p2_structured_rolling_bootstrap.csv"); RY = pd.read_csv(TAB/"p2_structured_rolling_by_year.csv")
main = pd.read_csv(TAB/"p2_main_summary_test.csv"); D = pd.read_csv(ROOT/"data/processed/phase2_lag_features.csv.gz"); parts = split_frame(D)
base = np.abs(parts["test"][TARGET] - parts["test"][BASE_COL]).mean()
T = M[M.split == "test"].groupby(["feature_set", "model"]).MAE.agg(["mean", "std"]).reset_index()
V = M[M.split == "validation"].groupby(["feature_set", "model"]).MAE.mean().unstack().round(1); print("VALIDATION MAE (selection data):"); V
"""),
code(r"""
t = T.pivot(index="feature_set", columns="model", values="mean")
f5 = main[(main.tier == "C_anomaly") & (main.feature_set == "F5_Lag16")].set_index("model").MAE
t.loc["F5 16 free lags (notebook 07)"] = f5; t = t.sort_values("RF"); print("TEST MAE, kg/ha (5 seeds); no-weather baseline = %.1f" % base); t.round(1)
"""),
code(r"""
fig, ax = plt.subplots(figsize=(11, 4.6)); order = ["F5 16 free lags (notebook 07)", "F8_Total_Anomaly (reference)", "S4_AsymTotal", "S3_StageAnomaly", "S1_Smooth2", "S2_Smooth3", "S7_StageAnomaly+Asym"]
pal = {"Ridge": "#8d99ae", "RF": "#2a9d8f", "XGB": "#e76f51"}
for k, mod in enumerate(["Ridge", "RF", "XGB"]):
    ax.bar(np.arange(len(order)) + (k - 1) * .27, t.loc[order, mod], .27, color=pal[mod], label=mod)
ax.axhline(base, color="k", ls=":", label="no-weather baseline"); ax.set_ylim(322, 342); ax.set_xticks(range(len(order)))
ax.set_xticklabels([o.replace("_", "\n", 1) for o in order], fontsize=8); ax.set_ylabel("Test MAE (kg/ha)"); ax.legend(frameon=False, ncol=4)
ax.set_title("Structured lag representations vs free lags and the seasonal total (tier C, test 2014-2017)"); savefig("p2_12_structured_lags"); plt.show()
"""),
md("## Validation-based selection and paired tests against the seasonal total"),
code(r"""
print("Candidate chosen per model on VALIDATION MAE:"); display(SEL.round(2)) if False else None
SEL.round(2)
"""),
code(r"""
B["significant_after_Holm"] = B.p_holm < .05
B.round(3)
"""),
md("## Robustness: rolling-origin one-year-ahead forecasts 2011-2017 (hyper-parameters fixed)"),
code(r"""
print(RB.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(8.5, 4)); 
for fs, c in [("F5_Lag16", "#d62828"), ("F8_Total_Anomaly", "#264653"), ("S1_Smooth2", "#2a9d8f"), ("S2_Smooth3", "#8ab17d")]:
    q = RY[(RY.model == "RF") & (RY.feature_set == fs)]; ax.plot(q.test_year, q.ae, marker="o", color=c, label=fs)
ax.set_xlabel("forecast year"); ax.set_ylabel("MAE (kg/ha), RF"); ax.legend(frameon=False, fontsize=8); ax.set_title("Rolling-origin RF by forecast year")
savefig("p2_12_structured_rolling"); plt.show()
"""),
md(r"""
### Reading
* **Structure rescues the lags.** Smooth lag profiles (3-4 parameters) are 5-6 kg/ha (RF) and 8 kg/ha (XGB) better than 16 free lags, and they match or slightly beat the seasonal total in all three model families.
* **The gain over the total is small** (≈ 0.6 % on the main split, −2.0 to −4.1 kg/ha for RF/XGB), consistent in sign across smooth variants and models, but only the Ridge/degree-3 comparison survives Holm correction, and under rolling origin RF is −1.1 [-2.4,+0.2] and XGB is null.
  We therefore describe it as *consistent but modest*, not as a decisive improvement.
* **Learned index and asymmetric split do not help** here; stage sums help only marginally.
"""),
]
if __name__ == "__main__":
    build("12_structured_lag_models", cells)
