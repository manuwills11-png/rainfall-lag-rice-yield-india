# Phase 2 data sources, extraction protocol and quality control

## Why a new data source was needed
Phase 1 (`docs/limitations.md` §1) established that `Data.csv` contains only seasonal aggregates, so the central proposal hypothesis
(16-week rainfall *timing*) could not be tested and synthetic lags are scientifically invalid. Phase 2 therefore adds **observed, time-resolved weather**.

| Variable | Product | Native resolution | Access | Script |
|---|---|---|---|---|
| Daily rainfall | IMD gridded rainfall (Pai et al.), 1901-present | 0.25° × 0.25°, daily | `imdlib` → imdpune.gov.in | `scripts/12_download_imd_grid.py` |
| Daily maximum temperature | IMD gridded Tmax | 1° × 1°, daily | `imdlib` | same |
| District location | Open-Meteo geocoder (district HQ), OSM Nominatim fallback, manual alias table for historical/mis-spelled names | point | public APIs | `scripts/10_geocode_districts.py`, `10b_fix_geocoding_aliases.py` |

IMD is the agency that produced the `IMD_RF`/`IMD_Tmax` columns of the yield dataset, so the new series is the most consistent
time-resolved counterpart available. (An alternative reanalysis source, ERA5 through the Open-Meteo API, was evaluated first but is rate-limited to a few
districts per minute; it was abandoned in favour of the IMD grid and is not used anywhere in the results.)

## Extraction protocol (`scripts/13_extract_district_weather.py`)
1. For each district point take the nearest IMD cell and the mean of the valid cells in the surrounding 3 × 3 block (≈ 75 km for rain).
2. Aggregate days to ISO weeks (rain: sum, Tmax: mean). 570,240 district-week rows, 0 missing.
3. Lag features: `Rain_Lag_k` = rainfall of ISO week `T_ref − k`, `k = 1…16`, `T_ref` from `src/crop_calendar.py`. Poisoning test (notebook 06 §4,
   `tests/test_weather_features.py`) shows no lag changes when all weather at/after `T_ref` is altered.

## Quality control and refinement (`scripts/14_refine_geolocation.py`)
* Pooled agreement between the extracted annual rainfall and the dataset's own `IMD_RF`: **r = 0.905** (N = 10,279); per-district median r = 0.917.
* Districts whose *training-year* (1990-2009) correlation was < 0.85 had their sampling block re-selected inside the surrounding 5 × 5 cell neighbourhood by maximum training-year correlation with `IMD_RF` (74 of 384 moved).
  Only the covariate `IMD_RF` and training years are used - never yield and never validation/test years.
* 13 districts keep r < 0.7 and are listed in `results/tables/p2_flagged_low_qc_districts.csv`; a sensitivity analysis (notebook 08 §6) excludes them.

## Known weaknesses of the weather data
* One point per district, not a polygon average (no district shapefile is part of the supplied data).
* 1° Tmax is coarse; temperature-lag results are treated as exploratory.
* Dataset districts follow historical boundaries; some geocoded names map to successor districts.
