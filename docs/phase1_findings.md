# Phase 1 Empirical Findings: India-wide Rice Yield Baseline

**Project:** Rainfall-Lag-Aware Crop Yield Forecasting in India Using Explainable Temporal Machine Learning  
**Date:** September 2026  
**Status:** Executed and Verified on Actual Historical Dataset

---

## 1. Executive Summary of Observed Metrics

The Phase 1 baseline pipeline was executed across 10,279 district-year observations (1990–2017) using a strictly chronological split (Train: 1990–2009, Validation: 2010–2013, Test: 2014–2017). All reported metrics derive strictly from executed code and predictions stored in `results/tables/phase1_metrics.csv`.

### Primary Model Benchmark Table:

| Model | Split | Observations ($N$) | MAE ($\text{kg/ha}$) | RMSE ($\text{kg/ha}$) | $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **Train (1990–2009)** | 7,361 | 276.574 | 370.373 | 0.8351 |
| **Random Forest** | **Validation (2010–2013)** | 1,448 | 606.299 | 774.581 | 0.3386 |
| **Random Forest** | **Test (2014–2017)** | **1,470** | **727.194** | **914.197** | **0.1333** |
| **XGBoost** | **Train (1990–2009)** | 7,361 | 311.077 | 415.272 | 0.7926 |
| **XGBoost** | **Validation (2010–2013)** | 1,448 | 601.744 | 772.979 | 0.3414 |
| **XGBoost** | **Test (2014–2017)** | **1,470** | **732.229** | **921.292** | **0.1198** |

---

## 2. Comparative Model Analysis

1. **Random Forest vs. XGBoost on Test Partition:**
   - Random Forest achieved an observed Test MAE of **727.19 kg/ha** ($\text{RMSE} = 914.20\text{ kg/ha}, R^2 = 0.1333$).
   - XGBoost achieved an observed Test MAE of **732.23 kg/ha** ($\text{RMSE} = 921.29\text{ kg/ha}, R^2 = 0.1198$).
   - The performance difference between Random Forest and XGBoost is modest ($\Delta \text{MAE} = 5.03\text{ kg/ha}$ or $0.69\%$), indicating that both bagging and gradient boosting models extract comparable predictive structure from aggregate seasonal weather totals.

2. **Generalization Gap Across Chronological Splits:**
   - Both models display a noticeable degradation from Train (MAE $\approx 276–311\text{ kg/ha}$, $R^2 \approx 0.79–0.84$) to Validation (MAE $\approx 601–606\text{ kg/ha}$, $R^2 \approx 0.34$) and Test (MAE $\approx 727–732\text{ kg/ha}$, $R^2 \approx 0.12–0.13$).
   - This performance drop highlights the non-stationary nature of temporal agricultural systems: climatic shifts, evolving seed varieties, expansion of irrigation infrastructure, and macroeconomic changes over the 28-year span alter the mapping from aggregate climate variables to realized yields.

---

## 3. Disaggregated Spatial Error Analysis

Test error exhibits substantial spatial variation across the 20 states (`results/tables/state_test_metrics.csv`):

### Top 5 Lowest Observed Test MAE States:
1. **Himachal Pradesh:** $\text{MAE} = 389.54\text{ kg/ha}$ ($\text{RMSE} = 493.10\text{ kg/ha}, N = 44$)
2. **Uttarakhand:** $\text{MAE} = 418.34\text{ kg/ha}$ ($\text{RMSE} = 601.87\text{ kg/ha}, N = 36$)
3. **Chhattisgarh:** $\text{MAE} = 485.49\text{ kg/ha}$ ($\text{RMSE} = 622.33\text{ kg/ha}, N = 28$)
4. **Uttar Pradesh:** $\text{MAE} = 492.04\text{ kg/ha}$ ($\text{RMSE} = 583.49\text{ kg/ha}, N = 216$)
5. **Kerala:** $\text{MAE} = 492.29\text{ kg/ha}$ ($\text{RMSE} = 593.18\text{ kg/ha}, N = 56$)

### Top 5 Highest Observed Test MAE States:
1. **Tamil Nadu:** $\text{MAE} = 1193.28\text{ kg/ha}$ ($\text{RMSE} = 1382.67\text{ kg/ha}, N = 80$)
2. **Bihar:** $\text{MAE} = 1182.96\text{ kg/ha}$ ($\text{RMSE} = 1382.25\text{ kg/ha}, N = 118$)
3. **Madhya Pradesh:** $\text{MAE} = 1058.67\text{ kg/ha}$ ($\text{RMSE} = 1344.72\text{ kg/ha}, N = 140$)
4. **Rajasthan:** $\text{MAE} = 875.31\text{ kg/ha}$ ($\text{RMSE} = 989.10\text{ kg/ha}, N = 70$)
5. **Andhra Pradesh:** $\text{MAE} = 870.73\text{ kg/ha}$ ($\text{RMSE} = 1033.78\text{ kg/ha}, N = 52$)

### District Distribution:
- Median district-level test MAE across 378 evaluated districts is **628.7 kg/ha**.
- Districts with intensive groundwater irrigation or extreme climatic vulnerability deviate substantially from the national mean response curve.

---

## 4. Scientific Interpretation and Implications for Phase 2

1. **Failure of Aggregate Totals to Capture Intra-Seasonal Timing:**
   - An annual or seasonal aggregate total (`IMD_RF`) cannot distinguish whether 1000 mm of rain fell evenly during vegetative growth or as a catastrophic deluge during harvest.
   - The low test $R^2$ ($0.12–0.13$) confirms that aggregate meteorological variables alone are insufficient for accurate long-horizon district-level crop forecasting.

2. **Validation of the 16-Week Pre-Harvest Lag Hypothesis:**
   - These baseline empirical findings provide direct motivation for Phase 2.
   - Without time-resolved weekly rainfall sequences preceding harvest, models cannot detect critical phenological stress periods (e.g. panicle initiation, flowering, grain filling).

---

## 5. Update after Phase 2 (read this before citing Section 4)

Section 4 attributed the weak test performance to the lack of intra-seasonal rainfall timing. Phase 2 tested that with real daily IMD rainfall and **did not confirm it**:
the dominant cause is non-stationarity (test-year mean yield +32 % above training; a no-weather "mean of the last three seasons" forecast reaches R² = 0.756),
and 16 weekly lags do not outperform a seasonal total out of sample. See `docs/research_paper.md` (Sections 5-6).
