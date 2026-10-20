"""Build + execute notebook 09: LSTM / GRU sequence models with shuffled-week and no-weather controls."""
from nb_common import build, code, md

cells = [
md(r"""
# 09 · Deep sequence models: LSTM and GRU on the 16-week rainfall sequence

Architectures (`src/sequence_models.py`, specification in `docs/sequence_models_architecture.md`): a 2-layer LSTM or GRU encodes
$w_{-16}\to w_{-1}$; its final hidden state is concatenated with the static spatial vector (state one-hot + `IMD_Tmax`) and passed to an MLP head.
Training details (`src/sequence_experiment.py`): log1p-rainfall, training-year standardisation, standardised targets, AdamW (lr 1e-3, wd 1e-4),
gradient clipping 5.0, **early stopping on validation MAE only**, 60 epochs max, patience 10.

For every formulation tier we run three conditions with *identical architecture and static inputs*:

| condition | sequence input | what it tells us |
|---|---|---|
| `true_order` | the real 16 weekly rainfalls | the proposed model |
| `shuffled_weeks` | same values, random order inside each sample | does the **temporal ordering** matter to the recurrent network? |
| `no_weather` | zeros | how much of the performance is the weather at all (state + temperature + baseline only)? |

5 seeds for `true_order`, 3 seeds for the controls.
"""),
code(r"""
from src.experiment import *
Q = pd.read_csv(TAB/"p3_sequence_metrics.csv"); H = pd.read_csv(TAB/"p3_sequence_history.csv"); SP = pd.read_csv(PRED/"p3_sequence_test_predictions.csv.gz")
Mt = pd.read_csv(TAB/"p2_main_summary_test.csv")
D = pd.read_csv(ROOT/"data/processed/phase2_lag_features.csv.gz"); parts = split_frame(D)
print(Q[Q.split == "test"].groupby(["tier", "condition", "model"]).seed.nunique().unstack().to_string())
"""),
md("## 1 · Learning curves (true order, tier C)"),
code(r"""
h = H[(H.tier == "C_anomaly") & (H.condition == "true_order")]
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, arch in zip(axs, ["LSTM", "GRU"]):
    for seed, g in h[h.model == arch].groupby("seed"):
        ax.plot(g.epoch, g.val_mae * 1, alpha=.8, label=f"seed {seed}")
    ax.set_title(f"{arch}: validation MAE (standardised target units)"); ax.set_xlabel("epoch"); ax.set_ylabel("val MAE (z)")
axs[0].legend(frameon=False, fontsize=8); plt.tight_layout(); savefig("p2_09_learning_curves"); plt.show()
print(h.groupby(["model", "seed"]).epoch.max().groupby("model").agg(["mean", "min", "max"]).rename(columns={"mean": "mean epochs", "min": "min", "max": "max"}))
"""),
md("## 2 · Test performance by condition"),
code(r"""
T = Q[Q.split == "test"]
S = T.groupby(["tier", "condition", "model"]).agg(MAE=("MAE", "mean"), MAE_sd=("MAE", "std"), RMSE=("RMSE", "mean"), R2=("R2", "mean"), n=("seed", "nunique")).reset_index()
S.to_csv(TAB/"p3_sequence_summary_test.csv", index=False); S.round(3)
"""),
code(r"""
base_c = np.abs(parts["test"][TARGET] - parts["test"][BASE_COL]).mean()
tabC = {r.feature_set + "/" + r.model: r.MAE for r in Mt[(Mt.tier == "C_anomaly") & Mt.model.isin(["RF", "XGB", "Ridge"]) & Mt.feature_set.isin(["F1_Aggregate_IMDgrid", "F5_Lag16", "F8_Total_Anomaly", "F7_Lag16_Anomaly"])].itertuples()}
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
cols = {"true_order": "#264653", "shuffled_weeks": "#e9c46a", "no_weather": "#adb5bd"}
for ax, tier in zip(axs, ["C_anomaly", "A_weather_only"]):
    s = S[S.tier == tier]
    for k, c in enumerate(["true_order", "shuffled_weeks", "no_weather"]):
        q = s[s.condition == c].set_index("model").reindex(["LSTM", "GRU"])
        ax.bar(np.arange(2) + (k - 1) * .27, q.MAE, .27, yerr=q.MAE_sd.fillna(0), color=cols[c], label=c, capsize=3)
    if tier == "C_anomaly":
        ax.axhline(base_c, color="k", ls=":", label="3-season baseline"); ax.axhline(tabC["F8_Total_Anomaly/RF"], color="crimson", ls="--", label="best tabular (F8/RF)")
        ax.set_ylim(300, max(S[S.tier == tier].MAE) * 1.05)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["LSTM", "GRU"]); ax.set_ylabel("Test MAE (kg/ha)"); ax.set_title(tier)
axs[0].legend(frameon=False, fontsize=8); plt.tight_layout(); savefig("p2_09_sequence_conditions"); plt.show()
"""),
md("## 3 · Paired tests: does the true week order help the recurrent network?"),
code(r"""
rows = []
for tier in ["C_anomaly", "A_weather_only"]:
    for arch in ["LSTM", "GRU"]:
        def ens(cond):
            x = SP[(SP.tier == tier) & (SP.condition == cond) & (SP.model == arch)].groupby(["stable_geographic_id", "Year"]).agg(y=("Actual", "first"), p=("Predicted", "mean"))
            x["ae"] = (x.y - x.p).abs(); return x
        t = ens("true_order")
        for other in ["shuffled_weeks", "no_weather"]:
            o = ens(other); idx = t.index.intersection(o.index)
            rows.append(dict(tier=tier, model=arch, true_vs=other, **paired_cluster_bootstrap(t.loc[idx, "ae"].values, o.loc[idx, "ae"].values, np.array([i[0] for i in idx]))))
PT = pd.DataFrame(rows); PT.to_csv(TAB/"p3_sequence_paired_bootstrap.csv", index=False); PT
"""),
md("## 4 · Sequence models vs tabular models (same split, same tier)"),
code(r"""
rows = []
for r in S[S.tier == "C_anomaly"].itertuples(): rows.append(dict(model=f"{r.model} ({r.condition})", MAE=r.MAE, R2=r.R2))
for r in Mt[(Mt.tier == "C_anomaly") & Mt.feature_set.isin(["F1_Aggregate_IMDgrid", "F8_Total_Anomaly", "F5_Lag16", "F7_Lag16_Anomaly"])].itertuples(): rows.append(dict(model=f"{r.model} · {r.feature_set}", MAE=r.MAE, R2=r.R2))
rows.append(dict(model="No-weather 3-season baseline", MAE=base_c, R2=calculate_metrics(parts["test"][TARGET].values, parts["test"][BASE_COL].values)["R2"]))
CMP = pd.DataFrame(rows).sort_values("MAE"); CMP.to_csv(TAB/"p3_sequence_vs_tabular_tierC.csv", index=False); CMP.round(3).head(25)
"""),
md(r"""
### Take-aways of notebook 09
Read the paired-bootstrap table: a recurrent network that truly exploits *timing* must beat both its own shuffled-week and no-weather versions.
Differences versus the tabular models are reported alongside so the deep-learning claim of the proposal is evaluated against its strongest simple competitor.
"""),
]

if __name__ == "__main__":
    build("09_deep_sequence_models", cells)
