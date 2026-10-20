"""Build + execute notebook 10: explainability (TreeSHAP, permutation) and panel fixed-effects rainfall-response profile."""
from nb_common import build, code, md

cells = [
md(r"""
# 10 · Explainability and the rainfall-response profile

Proposal objective: *"an explainable characterisation of critical rainfall exposure windows"*. We triangulate three independent lenses on the
same 16 weekly lags so that no conclusion rests on one method:

1. **TreeSHAP** on the trained XGBoost model (what the *predictor* uses) - exact path-dependent TreeSHAP via `Booster.predict(pred_contribs=True)`.
2. **Permutation importance** on the held-out test years (what *hurts accuracy* when a week is scrambled).
3. **Panel fixed-effects distributed-lag regression** on `log(yield)` with district and year fixed effects, clustered by district
   (what the data say about the *effect* of a +100 mm anomaly in each week, with confidence intervals and joint tests).

> Caveat (also in `docs/limitations.md`): SHAP and permutation importance describe model behaviour; only the fixed-effects regression
> has a (conditional) causal-style reading, and even it assumes weather anomalies are as-good-as-random given the fixed effects.
"""),
code(r"""
import json
from src.experiment import *
from src.explain import *
from src.panel import lag_coefficients, stage_coefficients, fit_fe, _prep
from src.weather_features import STAGES
D = pd.read_csv(ROOT/"data/processed/phase2_lag_features.csv.gz"); D = add_rain_anomalies(D, 2009)
HP = json.load(open(TAB/"p2_main_hyperparameters.json"))
STAGE_COLOR = dict(zip(STAGES, ["#e9c46a", "#8ab17d", "#2a9d8f", "#e76f51", "#6d597a"]))
def stage_bands(ax, ymax=None):
    # lags are plotted at x = -k; stage k-sets are inclusive
    for (s, ks), c in zip(STAGES.items(), STAGE_COLOR.values()):
        ax.axvspan(-max(ks) - .5, -min(ks) + .5, color=c, alpha=.13, lw=0)
"""),
md("## 1 · Train the explained models and compute TreeSHAP on the held-out test years"),
code(r"""
parts = split_frame(D)
def fit_xgb(tier, fs):
    cols = FEATURE_SETS[fs]; mode = "anomaly" if tier == "C_anomaly" else "level"
    X, y, names, enc = design_matrices(parts, cols)
    base = {k: (v[BASE_COL].to_numpy() if mode == "anomaly" else np.zeros(len(v))) for k, v in parts.items()}
    p = HP[f"{tier}|{fs}|XGB"]; m = fit_model("XGB", p, X, {k: y[k] - base[k] for k in y}, 42)
    return m, X, {k: y[k] - base[k] for k in y}, names, base
models = {}
for key in [("C_anomaly", "F7_Lag16_Anomaly"), ("C_anomaly", "F5_Lag16"), ("A_weather_only", "F5_Lag16"), ("A_weather_only", "F7_Lag16_Anomaly")]:
    models[key] = fit_xgb(*key)
    m, X, y, names, base = models[key]; print(key, "test MAE (kg/ha):", round(np.mean(np.abs(y["test"] - m.predict(X["test"]))), 1))
"""),
code(r"""
prof = {}; shp = {}
for key, pref in [(("C_anomaly", "F7_Lag16_Anomaly"), "Rain_Anom_Lag_"), (("C_anomaly", "F5_Lag16"), "Rain_Lag_"), (("A_weather_only", "F5_Lag16"), "Rain_Lag_"), (("A_weather_only", "F7_Lag16_Anomaly"), "Rain_Anom_Lag_")]:
    m, X, y, names, base = models[key]
    sv = xgb_shap(m, X["test"], names); shp[key] = sv
    prof[key] = lag_profile(sv, prefix=pref)
    # additivity check: SHAP + bias reproduces the prediction
    chk = np.abs(sv.sum(axis=1).values - m.predict(X["test"])).max(); print(key, "max |ΣSHAP − prediction| =", f"{chk:.2e}")
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
for ax, key, ttl in zip(axs, [("C_anomaly", "F7_Lag16_Anomaly"), ("A_weather_only", "F5_Lag16")], ["Tier C · anomaly lags (F7)", "Tier A · raw lags (F5)"]):
    p = prof[key]; stage_bands(ax)
    ax.bar(p.weeks_before_harvest, p.mean_abs_shap, color=["#d62828" if s > 0 else "#1d3557" for s in p.mean_shap], width=.8)
    ax.set_xlabel("weeks before harvest reference week"); ax.set_ylabel("mean |SHAP| (kg/ha)"); ax.set_title(ttl)
    ax.set_xticks(range(-16, 0))
fig.suptitle("TreeSHAP rainfall-response profile (test years; red = higher rain raises prediction on average, blue = lowers)", y=1.03, fontweight="bold")
plt.tight_layout(); savefig("p2_10_shap_profile"); plt.show()
"""),
code(r"""
# SHAP dependence of the most important weeks (tier C, anomaly lags)
key = ("C_anomaly", "F7_Lag16_Anomaly"); m, X, y, names, base = models[key]; sv = shp[key]
top = prof[key].sort_values("mean_abs_shap", ascending=False).head(4)
fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, (_, r) in zip(axs, top.iterrows()):
    j = names.index(r.lag); ax.scatter(X["test"][:, j], sv[r.lag], s=5, alpha=.35, color="#264653")
    ax.axhline(0, color="k", lw=.8); ax.set_xlabel(f"rain anomaly, week {int(r.weeks_before_harvest)} (mm)"); ax.set_ylabel("SHAP (kg/ha)"); ax.set_title(f"week {int(r.weeks_before_harvest)}")
plt.tight_layout(); savefig("p2_10_shap_dependence_top_weeks"); plt.show()
"""),
md("## 2 · Stage-level view and group importance"),
code(r"""
sp = stage_profile(prof[("C_anomaly", "F7_Lag16_Anomaly")]); sp.to_csv(TAB/"p2_shap_stage_profile_tierC.csv", index=False)
sp2 = stage_profile(prof[("A_weather_only", "F5_Lag16")]); sp2.to_csv(TAB/"p2_shap_stage_profile_tierA.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 4))
w = .38; x = np.arange(len(sp))
ax.bar(x - w/2, sp.share_per_week * 100, w, label="Tier C (anomaly lags)", color="#2a9d8f"); ax.bar(x + w/2, sp2.share_per_week * 100, w, label="Tier A (raw lags)", color="#e9c46a")
ax.axhline(100 / 16, color="k", ls="--", lw=1, label="uniform (6.25 % per week)"); ax.set_xticks(x); ax.set_xticklabels([s.split("_", 1)[1].replace("_", "\n") for s in sp.stage])
ax.set_ylabel("share of lag-SHAP per week (%)"); ax.legend(frameon=False); ax.set_title("Importance per week, by phenological stage")
savefig("p2_10_stage_importance"); plt.show(); sp.round(3)
"""),
code(r"""
rows = []
for key in [("C_anomaly", "F7_Lag16_Anomaly"), ("A_weather_only", "F5_Lag16")]:
    m, X, y, names, base = models[key]
    pref = "Rain_Anom_Lag_" if "F7" in key[1] else "Rain_Lag_"
    groups = {"16 weekly rainfall lags (jointly)": [f"{pref}{k}" for k in range(1, 17)], "IMD_Tmax": ["IMD_Tmax"], "state fixed effects": [n for n in names if n.startswith("State_Clean_")]}
    gi = group_permutation_importance(m, X["test"], y["test"], names, groups, n_repeats=15); gi.insert(0, "tier", key[0]); rows.append(gi)
    wk = permutation_week_importance(m, X["test"], y["test"], names, n_repeats=15, prefix=pref); wk.to_csv(TAB/f"p2_perm_week_importance_{key[0]}.csv", index=False)
G = pd.concat(rows); G.to_csv(TAB/"p2_group_permutation_importance.csv", index=False); G
"""),
md(r"""
## 3 · Panel fixed-effects distributed-lag regression (explanatory lens)

$$\log y_{it} = \sum_{k=1}^{16}\beta_k\,\text{RainAnom}_{it,k} + \alpha_i + \gamma_t + \varepsilon_{it}$$

Units: $\beta_k\times 100$ = % change in yield per **+100 mm** of rain above the district's normal in week $-k$. District fixed effects absorb
time-invariant productivity, year fixed effects absorb all-India shocks and the technology trend.
"""),
code(r"""
res = lag_coefficients(D); tb = res["table"]; tb.to_csv(TAB/"p2_panel_fe_lag_coefficients.csv", index=False)
print(f"N = {res['n']:,} district-years · R² (with FE) = {res['r2']:.3f}")
print("H0: all 16 weekly coefficients = 0     :  Wald χ² = %.1f,  p = %.2e" % res["wald_zero"])
print("H0: all 16 coefficients EQUAL (timing irrelevant): Wald χ² = %.1f,  p = %.2e" % res["wald_equal"])
fig, ax = plt.subplots(figsize=(11, 4.6)); stage_bands(ax)
ax.errorbar(tb.weeks_before_harvest, tb.coef_pct_per_100mm, yerr=[tb.coef_pct_per_100mm - tb.ci_lo, tb.ci_hi - tb.coef_pct_per_100mm], fmt="o-", color="#1d3557", capsize=3)
ax.axhline(0, color="k", lw=1); ax.set_xticks(range(-16, 0)); ax.set_xlabel("weeks before harvest reference week"); ax.set_ylabel("yield change per +100 mm (%)")
ax.set_title("Panel FE distributed-lag response (95% CI, district-clustered)")
savefig("p2_10_panel_fe_profile"); plt.show(); tb.round(3)
"""),
code(r"""
st = stage_coefficients(D); st.to_csv(TAB/"p2_panel_fe_stage_coefficients.csv", index=False)
d_ = _prep(D); fe_only = fit_fe(d_, ["IMD_Tmax"], "C(dist) + C(Year)")
mt = fit_fe(d_.assign(tot=D.Rain_Anom_Total16 / 100), ["tot"]); ml = res["model"]
r2 = {"district+year FE (+Tmax)": fe_only.rsquared, "+ seasonal total anomaly": mt.rsquared, "+ 16 weekly anomalies": ml.rsquared}
within = lambda r: (r - fe_only.rsquared) / (1 - fe_only.rsquared)
print({k: round(v, 4) for k, v in r2.items()})
print("share of the FE-residual variance explained by the seasonal total  : %.2f %%" % (100 * within(mt.rsquared)))
print("share of the FE-residual variance explained by the 16 weekly lags  : %.2f %%" % (100 * within(ml.rsquared)))
pd.DataFrame([dict(model=k, R2=v, incremental_within_R2_pct=100 * within(v)) for k, v in r2.items()]).to_csv(TAB/"p2_panel_fe_r2_decomposition.csv", index=False)
st.round(3)
"""),
md("## 4 · Do the three lenses agree?"),
code(r"""
from scipy.stats import spearmanr, pearsonr
a = prof[("C_anomaly", "F7_Lag16_Anomaly")].set_index("weeks_before_harvest").mean_abs_shap
b = pd.read_csv(TAB/"p2_perm_week_importance_C_anomaly.csv").set_index("weeks_before_harvest").mae_increase
c = tb.set_index("weeks_before_harvest").coef_pct_per_100mm.abs()
T = pd.DataFrame({"TreeSHAP |mean|": a, "permutation ΔMAE": b, "panel |β|": c})
print(T.corr(method="spearman").round(2))
sgn = (np.sign(prof[("C_anomaly", "F7_Lag16_Anomaly")].set_index("weeks_before_harvest").mean_shap) == np.sign(tb.set_index("weeks_before_harvest").coef_pct_per_100mm)).mean()
print("sign agreement between mean SHAP and panel β across the 16 weeks:", round(float(sgn), 2))
Z = (T - T.min()) / (T.max() - T.min())
fig, ax = plt.subplots(figsize=(10.5, 4));
for col, mk in zip(Z.columns, ["o-", "s--", "^:"]): ax.plot(Z.index, Z[col], mk, label=col)
stage_bands(ax); ax.set_xticks(range(-16, 0)); ax.set_ylabel("min-max scaled importance"); ax.set_xlabel("weeks before harvest"); ax.legend(frameon=False); ax.set_title("Triangulation of the rainfall-response profile")
savefig("p2_10_triangulation"); plt.show()
"""),
md("## 5 · Is the response non-linear? And does it differ between irrigated and rain-fed states?"),
code(r"""
d_ = _prep(D).copy(); d_["tot"] = D.Rain_Anom_Total16
# residualise log yield on district + year FE, then bin by total anomaly (in district-normal units)
mfe = fit_fe(d_, ["IMD_Tmax"], "C(dist) + C(Year)"); d_["resid"] = mfe.resid
d_["tot_bin"] = pd.qcut(d_.tot, 12); bm = d_.groupby("tot_bin").agg(x=("tot", "mean"), y=("resid", "mean"), se=("resid", lambda s: s.std() / np.sqrt(len(s)))).reset_index()
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
axs[0].errorbar(bm.x, bm.y * 100, yerr=bm.se * 196, fmt="o-", color="#264653", capsize=3); axs[0].axhline(0, color="k", lw=.8); axs[0].axvline(0, color="k", lw=.8, ls=":")
axs[0].set_xlabel("16-week rainfall anomaly (mm vs district normal)"); axs[0].set_ylabel("yield residual after FE (%)"); axs[0].set_title("Non-linearity of the seasonal rainfall response")
IRR = {"Punjab", "Haryana", "Uttar Pradesh", "Andhra Pradesh", "Telangana", "Tamil Nadu"}
for name, mask, col in [("predominantly irrigated", D.State_Clean.isin(IRR), "#e9c46a"), ("rain-fed / mixed", ~D.State_Clean.isin(IRR), "#2a9d8f")]:
    r_ = lag_coefficients(D[mask]); t_ = r_["table"]
    axs[1].errorbar(t_.weeks_before_harvest + (.12 if "rain" in name else -.12), t_.coef_pct_per_100mm, yerr=[t_.coef_pct_per_100mm - t_.ci_lo, t_.ci_hi - t_.coef_pct_per_100mm], fmt="o-", color=col, capsize=2, label=f"{name} (N={r_['n']:,})")
    print(name, "| equal-coefficient test p = %.2e | zero test p = %.2e" % (r_["wald_equal"][1], r_["wald_zero"][1]))
    t_.assign(group=name).to_csv(TAB/f"p2_panel_fe_{name.split()[0].replace('/', '')}.csv", index=False)
axs[1].axhline(0, color="k", lw=.8); axs[1].set_xticks(range(-16, 0)); axs[1].set_xlabel("weeks before harvest"); axs[1].set_ylabel("% per +100 mm"); axs[1].legend(frameon=False, fontsize=8)
axs[1].set_title("Lag response by irrigation regime"); plt.tight_layout(); savefig("p2_10_nonlinearity_and_regimes"); plt.show()
"""),
md(r"""
### Take-aways of notebook 10
The explanatory regression and the model-based importances can be compared directly (table above). The key distinction the project must
keep clear: a **statistically structured** rainfall-timing response (explanatory) is not the same thing as an **out-of-sample forecasting gain**
(notebooks 07-09); notebook 11 reconciles the two.
"""),
]

if __name__ == "__main__":
    build("10_explainability_and_rainfall_response", cells)
