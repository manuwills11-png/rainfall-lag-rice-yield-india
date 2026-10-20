"""Extract district-level daily IMD weather and aggregate to ISO weeks.

For every district point (lat, lon) the daily value is the mean over the valid cells of the 3x3 block of IMD grid
cells centred on the nearest cell (rain: 0.25 deg => ~75 km box; Tmax: 1 deg native grid, nearest-cell block of 3x3).
Output: data/external/district_weekly_weather.csv.gz [stable_geographic_id, iso_year, iso_week, rain_mm, tmax_c, n_days]
        data/external/district_annual_weather_qc.csv     (calendar-year totals for QC against IMD_RF)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import imdlib as imd

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "data" / "external" / "imd_grid"
COORDS = ROOT / "data" / "external" / "district_coordinates_final.csv"


def block_mean_series(arr, lat_idx, lon_idx, k=1):
    """arr (T, nlat, nlon) -> (T,) nanmean over a (2k+1)^2 block around each point; one series per point."""
    out = []
    for i, j in zip(lat_idx, lon_idx):
        blk = arr[:, max(i - k, 0): i + k + 1, max(j - k, 0): j + k + 1].reshape(arr.shape[0], -1)
        with np.errstate(all="ignore"):
            out.append(np.nanmean(blk, axis=1))
    return np.vstack(out)       # (n_points, T)


def main():
    pts = pd.read_csv(COORDS).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    weekly_frames, annual_rows = [], []
    for yr in range(1990, 2018):
        series = {}
        for var, col in (("rain", "rain_mm"), ("tmax", "tmax_c")):
            d = imd.open_data(var, yr, yr, "yearwise", str(GRID))
            ds = d.get_xarray()
            name = list(ds.data_vars)[0]
            arr = ds[name].values.astype(float)               # (T, nlat, nlon)
            lats, lons = ds["lat"].values, ds["lon"].values
            li = np.abs(lats[None, :] - pts["lat"].values[:, None]).argmin(1)
            lj = np.abs(lons[None, :] - pts["lon"].values[:, None]).argmin(1)
            series[col] = block_mean_series(arr, li, lj, k=1)  # (n_pts, T)
            dates = pd.to_datetime(ds["time"].values)
        iso = dates.isocalendar()
        for p, geo in enumerate(pts["stable_geographic_id"]):
            df = pd.DataFrame({"iso_year": iso["year"].values, "iso_week": iso["week"].values,
                               "rain_mm": series["rain_mm"][p], "tmax_c": series["tmax_c"][p]})
            w = df.groupby(["iso_year", "iso_week"]).agg(rain_mm=("rain_mm", "sum"), tmax_c=("tmax_c", "mean"),
                                                         n_days=("rain_mm", "size")).reset_index()
            w.insert(0, "stable_geographic_id", geo)
            weekly_frames.append(w)
            annual_rows.append(dict(stable_geographic_id=geo, Year=yr, grid_rain_annual=np.nansum(series["rain_mm"][p]),
                                    grid_rain_jun_nov=np.nansum(series["rain_mm"][p][(dates.month >= 6) & (dates.month <= 11)]),
                                    grid_tmax_mean=np.nanmean(series["tmax_c"][p]),
                                    n_nan_rain_days=int(np.isnan(series["rain_mm"][p]).sum())))
        print("done", yr, flush=True)
    wk = pd.concat(weekly_frames, ignore_index=True)
    wk = wk[(wk.iso_year >= 1990) & (wk.iso_year <= 2017)]
    wk.to_csv(ROOT / "data" / "external" / "district_weekly_weather.csv.gz", index=False)
    pd.DataFrame(annual_rows).to_csv(ROOT / "data" / "external" / "district_annual_weather_qc.csv", index=False)


if __name__ == "__main__":
    main()
