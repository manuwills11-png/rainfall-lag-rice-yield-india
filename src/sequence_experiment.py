"""Training / evaluation harness for the LSTM and GRU models on the 16-week rainfall sequence.

Differences w.r.t. the raw ``SequenceTrainer`` defaults (kept in ``src.sequence_models``):
* rainfall is log1p-transformed and standardised with *training-year* statistics;
* the target is standardised with training-year statistics (raw kg/ha targets make MSE training unstable);
* early stopping / checkpoint selection use validation MAE only; the test years are predicted once at the end.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.evaluation import calculate_metrics
from src.experiment import BASE_COL, CAT, TARGET, split_frame
from src.sequence_models import CropYieldSequenceDataset, RiceGRU, RiceLSTM, SequenceTrainer
from src.weather_features import RAIN_LAGS, TMAX_LAGS


def _seq_tensor(df: pd.DataFrame, channels: Sequence[Sequence[str]]) -> np.ndarray:
    """(N, 16, C): one column group per channel, each already in chronological order."""
    return np.stack([df[list(cols)].to_numpy(float) for cols in channels], axis=-1)


def run_sequence(
    data: pd.DataFrame,
    arch: str,
    seed: int,
    channels: Sequence[Sequence[str]] = (RAIN_LAGS,),
    static_num: Sequence[str] = ("IMD_Tmax",),
    hidden_dim: int = 64,
    max_epochs: int = 60,
    patience: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    shuffle_weeks: bool = False,
    target_mode: str = "level",
    zero_weather: bool = False,
) -> Dict:
    """Train one sequence model. Returns dict(preds=DataFrame, metrics=list, history=DataFrame)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    parts = split_frame(data)

    # --- sequence channel scaling (train statistics, log1p for rainfall channels)
    seqs = {k: _seq_tensor(v, channels) for k, v in parts.items()}
    for c, cols in enumerate(channels):
        is_rain = cols[0].startswith("Rain")
        if is_rain:
            for k in seqs:
                seqs[k][..., c] = np.log1p(seqs[k][..., c])
        mu, sd = seqs["train"][..., c].mean(), seqs["train"][..., c].std() + 1e-8
        for k in seqs:
            seqs[k][..., c] = (seqs[k][..., c] - mu) / sd

    if zero_weather:      # control: identical architecture / static inputs, no weather information at all
        seqs = {k: np.zeros_like(v) for k, v in seqs.items()}

    if shuffle_weeks:     # control: independent random week order inside every sample (aggregates preserved)
        rng = np.random.default_rng(1234)
        for k in seqs:
            for i in range(len(seqs[k])):
                seqs[k][i] = seqs[k][i][rng.permutation(seqs[k].shape[1])]

    # --- static: state one-hot (fit on train) + scaled numeric
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(parts["train"][CAT])
    stat = {}
    for k, v in parts.items():
        parts_stat = [enc.transform(v[CAT])]
        if static_num:
            parts_stat.append(v[list(static_num)].to_numpy(float))
        stat[k] = np.hstack(parts_stat)
    if static_num:
        n_oh = enc.transform(parts["train"][CAT]).shape[1]
        mu = stat["train"][:, n_oh:].mean(0); sd = stat["train"][:, n_oh:].std(0) + 1e-8
        for k in stat:
            stat[k][:, n_oh:] = (stat[k][:, n_oh:] - mu) / sd

    # --- target scaling
    y = {k: v[TARGET].to_numpy(float) for k, v in parts.items()}
    base = {k: (v[BASE_COL].to_numpy(float) if target_mode == "anomaly" else np.zeros(len(v))) for k, v in parts.items()}
    dy = {k: y[k] - base[k] for k in y}
    ym, ys = dy["train"].mean(), dy["train"].std()
    yz = {k: (v - ym) / ys for k, v in dy.items()}

    ds = {k: CropYieldSequenceDataset(seqs[k], stat[k], yz[k]) for k in parts}
    g = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True, generator=g)
    loaders = {k: DataLoader(ds[k], batch_size=512, shuffle=False) for k in ("train", "validation", "test")}

    cls = RiceLSTM if arch == "LSTM" else RiceGRU
    model = cls(seq_input_dim=len(channels), static_input_dim=stat["train"].shape[1], hidden_dim=hidden_dim, num_layers=2, dropout=0.2)
    trainer = SequenceTrainer(model, lr=lr, weight_decay=1e-4, patience=patience)
    hist = trainer.fit(tr_loader, loaders["validation"], max_epochs=max_epochs)

    out_pred, mets = [], []
    for split in ("train", "validation", "test"):
        yh = trainer.predict(loaders[split]) * ys + ym + base[split]
        mets.append(dict(model=arch, seed=seed, split=split, **calculate_metrics(y[split], yh), n=len(yh)))
        if split != "train":
            d = parts[split][["stable_geographic_id", "State_Clean", "Year"]].copy()
            d.insert(0, "split", split); d.insert(0, "seed", seed); d.insert(0, "model", arch)
            d["Actual"] = y[split]; d["Predicted"] = yh
            out_pred.append(d)
    return dict(preds=pd.concat(out_pred, ignore_index=True), metrics=mets, history=pd.DataFrame(hist))
