"""Train LSTM / GRU on the 16-week rainfall sequence (+ shuffled-week and no-weather controls). Resumable.

Output: results/tables/p3_sequence_metrics.csv, results/predictions/p3_sequence_test_predictions.csv.gz,
        results/tables/p3_sequence_history.csv
"""
import sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
torch.set_num_threads(4)
from src.sequence_experiment import run_sequence

D = pd.read_csv(ROOT / "data/processed/phase2_lag_features.csv.gz")
OUT = ROOT / "results/tables/p3_sequence_metrics.csv"
PRED = ROOT / "results/predictions/p3_sequence_test_predictions.csv.gz"
HIST = ROOT / "results/tables/p3_sequence_history.csv"
SEEDS5, SEEDS3 = (42, 7, 123, 2024, 99), (42, 7, 123)
done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["tier", "condition", "model", "seed"])
preds = [pd.read_csv(PRED)] if PRED.exists() else []
hists = [pd.read_csv(HIST)] if HIST.exists() else []
mets = [done] if len(done) else []
key = set(zip(done.tier, done.condition, done.model, done.seed))
t0 = time.time()
for tier, tm in (("C_anomaly", "anomaly"), ("A_weather_only", "level")):
    for cond, kw, seeds in (("true_order", {}, SEEDS5), ("shuffled_weeks", dict(shuffle_weeks=True), SEEDS3), ("no_weather", dict(zero_weather=True), SEEDS3)):
        for arch in ("LSTM", "GRU"):
            for seed in seeds:
                if (tier, cond, arch, seed) in key:
                    continue
                r = run_sequence(D, arch, seed, target_mode=tm, **kw)
                m = pd.DataFrame(r["metrics"]); m.insert(0, "condition", cond); m.insert(0, "tier", tier)
                p = r["preds"].copy(); p.insert(0, "condition", cond); p.insert(0, "tier", tier)
                h = r["history"].copy(); h.insert(0, "seed", seed); h.insert(0, "model", arch); h.insert(0, "condition", cond); h.insert(0, "tier", tier)
                mets.append(m); preds.append(p[p.split == "test"]); hists.append(h)
                pd.concat(mets).to_csv(OUT, index=False); pd.concat(preds).to_csv(PRED, index=False); pd.concat(hists).to_csv(HIST, index=False)
                t = m[m.split == "test"].iloc[0]
                print(f"{tier} {cond} {arch} seed={seed} epochs={len(r['history'])} test MAE={t.MAE:.1f} R2={t.R2:.3f} ({time.time()-t0:.0f}s)", flush=True)
print("ALL DONE")
