"""Run the full tabular experiment suite and write every result table to results/tables/ (resumable).

Stages (each skipped when its output exists, unless --force):
  E1  main ablation      : 9 feature sets x 3 formulation tiers x {Ridge, RF, XGB}, 5 seeds
  E2a shuffle control    : F5 / F7 with week order randomised inside each row
  E2b anchor sweep       : placebo windows moved 0..16 weeks earlier
  E2c window length      : last L lags only
  E7  rolling origin     : expanding-window one-year-ahead forecasts 2011-2017
Run:  python scripts/20_run_tabular_experiments.py [--force]
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.experiment import *                                   # noqa
from src.weather_features import build_lag_table, season_summary, RAIN_LAGS
from src.experiment import add_context_features

FORCE = "--force" in sys.argv
SEEDS = (42, 7, 123, 2024, 99)
TAB = ROOT / "results" / "tables"; PRED = ROOT / "results" / "predictions"
TAB.mkdir(parents=True, exist_ok=True); PRED.mkdir(parents=True, exist_ok=True)
TIERS = {"A_weather_only": dict(target_mode="level", ctx=False),
         "B_level_plus_context": dict(target_mode="level", ctx=True),
         "C_anomaly": dict(target_mode="anomaly", ctx=False)}


def todo(path):
    return FORCE or not Path(path).exists()


def load():
    D = pd.read_csv(ROOT / "data/processed/phase2_lag_features.csv.gz")
    return add_rain_anomalies(D, train_end=2009)


def e1(D):
    if not todo(TAB / "p2_main_metrics.csv"):
        return
    allm, allp, allparams = [], [], {}
    t0 = time.time()
    for tier, cfg in TIERS.items():
        for fs, cols in FEATURE_SETS.items():
            cols = cols + (CTX if cfg["ctx"] else [])
            p, m, params, _ = run_tabular(D, fs, cols, seeds=SEEDS, target_mode=cfg["target_mode"])
            p.insert(0, "tier", tier); m.insert(0, "tier", tier)
            allp.append(p[p.split == "test"]); allm.append(m)
            for (f, mod), v in params.items():
                allparams[f"{tier}|{f}|{mod}"] = v
            print(f"[E1] {tier} {fs} done ({time.time()-t0:.0f}s)", flush=True)
    pd.concat(allm).to_csv(TAB / "p2_main_metrics.csv", index=False)
    pd.concat(allp).to_csv(PRED / "p2_main_test_predictions.csv.gz", index=False)
    json.dump(allparams, open(TAB / "p2_main_hyperparameters.json", "w"), indent=1)
    # reference rows
    naive_baselines(D).to_csv(TAB / "p2_naive_baselines.csv", index=False)


def e2a(D):
    if not todo(TAB / "p2_control_shuffle_metrics.csv"):
        return
    params = {tuple(k.split("|")[1:]): v for k, v in json.load(open(TAB / "p2_main_hyperparameters.json")).items() if k.startswith("C_anomaly|")}
    rows = []
    for name, cols in (("F5_Lag16", RAIN_LAGS), ("F7_Lag16_Anomaly", ANOM_LAGS)):
        Ds = shuffle_within_row(D, cols, seed=0)
        _, m, _, _ = run_tabular(Ds, name + "__ShuffledWeeks", FEATURE_SETS[name], models=("RF", "XGB"), seeds=SEEDS,
                                 tuned={(name + "__ShuffledWeeks", mod): params[(name, mod)] for mod in ("RF", "XGB")}, target_mode="anomaly")
        rows.append(m)
        print("[E2a]", name, flush=True)
    pd.concat(rows).to_csv(TAB / "p2_control_shuffle_metrics.csv", index=False)


def e2b(D0):
    if not todo(TAB / "p2_control_anchor_shift_metrics.csv"):
        return
    clean = pd.read_csv(ROOT / "data/processed/phase1_clean.csv")
    weekly = pd.read_csv(ROOT / "data/external/district_weekly_weather.csv.gz")
    params = {tuple(k.split("|")[1:]): v for k, v in json.load(open(TAB / "p2_main_hyperparameters.json")).items() if k.startswith("C_anomaly|")}
    rows = []
    for shift in (0, 2, 4, 6, 8, 12, 16):
        L = build_lag_table(clean, weekly, anchor_shift=shift, include_tmax=False)
        L = add_context_features(L)
        L = add_rain_anomalies(L, train_end=2009)
        for name in ("F5_Lag16", "F7_Lag16_Anomaly"):
            _, m, _, _ = run_tabular(L, f"{name}@shift{shift}", FEATURE_SETS[name].copy() if name != "F5_Lag16" else RAIN_LAGS + ["IMD_Tmax"],
                                     models=("RF", "XGB"), seeds=SEEDS[:3],
                                     tuned={(f"{name}@shift{shift}", mod): params[(name, mod)] for mod in ("RF", "XGB")}, target_mode="anomaly")
            m["anchor_shift_weeks"] = shift; m["base_feature_set"] = name
            rows.append(m)
        print("[E2b] shift", shift, flush=True)
    pd.concat(rows).to_csv(TAB / "p2_control_anchor_shift_metrics.csv", index=False)


def e2c(D):
    if not todo(TAB / "p2_window_length_metrics.csv"):
        return
    params = {tuple(k.split("|")[1:]): v for k, v in json.load(open(TAB / "p2_main_hyperparameters.json")).items() if k.startswith("C_anomaly|")}
    rows = []
    for Lw in (2, 4, 6, 8, 10, 12, 14, 16):
        cols = RAIN_LAGS[-Lw:] + ["IMD_Tmax"]
        _, m, _, _ = run_tabular(D, f"Lag{Lw}", cols, models=("RF", "XGB"), seeds=SEEDS[:3],
                                 tuned={(f"Lag{Lw}", mod): params[("F5_Lag16", mod)] for mod in ("RF", "XGB")}, target_mode="anomaly")
        m["window_weeks"] = Lw; rows.append(m)
        print("[E2c] window", Lw, flush=True)
    pd.concat(rows).to_csv(TAB / "p2_window_length_metrics.csv", index=False)


def e7(D0):
    if not todo(TAB / "p2_rolling_origin_predictions.csv.gz"):
        return
    allp = json.load(open(TAB / "p2_main_hyperparameters.json"))
    out = []
    for tier, cfg in (("A_weather_only", TIERS["A_weather_only"]), ("C_anomaly", TIERS["C_anomaly"])):
        for fs in ("F0_Aggregate_IMD", "F1_Aggregate_IMDgrid", "F5_Lag16", "F7_Lag16_Anomaly", "F8_Total_Anomaly"):
            for model in ("RF", "XGB"):
                r = rolling_origin_eval(D0, fs, FEATURE_SETS[fs], list(range(2011, 2018)), model, allp[f"{tier}|{fs}|{model}"],
                                        target_mode=cfg["target_mode"], seeds=SEEDS[:3])
                r.insert(0, "tier", tier); out.append(r)
            print("[E7]", tier, fs, flush=True)
    pd.concat(out).to_csv(TAB / "p2_rolling_origin_predictions.csv.gz", index=False)


if __name__ == "__main__":
    D = load()
    D0 = pd.read_csv(ROOT / "data/processed/phase2_lag_features.csv.gz")
    for stage in (e1, e2a, e2b, e2c, e7):
        stage(D if stage in (e1, e2a, e2c) else D0 if stage is e7 else D)
    print("ALL DONE")
