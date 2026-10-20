"""QC-driven refinement of district sampling points.

District points from the geocoders are administrative-HQ towns, which can sit far from the district's rain-weighted
centre (hill districts, coastal strips) or be a wrong homonym. For districts whose *training-year* (1990-2009) annual
IMD-grid rainfall correlates poorly with the dataset's own ``IMD_RF`` column, we search the 5x5 neighbourhood of grid
cells around the geocoded point and keep the 3x3 block with the highest training-year correlation.

Only the covariate ``IMD_RF`` and training years are used -- never yield, and never validation/test years.
Outputs: data/external/district_coordinates_final.csv (overwritten; keeps lat0/lon0 and QC columns).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import imdlib as imd

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "data" / "external" / "imd_grid"
CO = ROOT / "data" / "external" / "district_coordinates_final.csv"
CACHE = ROOT / "data" / "external" / "annual_cell_sums.npz"


def annual_cell_sums():
    if CACHE.exists():
        z = np.load(CACHE); return z["sums"], z["lats"], z["lons"], z["years"]
    years = np.arange(1990, 2018); sums = []
    for yr in years:
        ds = imd.open_data("rain", int(yr), int(yr), "yearwise", str(GRID)).get_xarray()
        a = ds["rain"].values.astype(float)
        sums.append(np.nansum(a, axis=0) + 0 * np.nanmax(a, axis=0))     # NaN cells stay NaN via the max term
        lats, lons = ds["lat"].values, ds["lon"].values
    sums = np.stack(sums)
    np.savez(CACHE, sums=sums, lats=lats, lons=lons, years=years)
    return sums, lats, lons, years


def block(sums, i, j, k=1):
    return np.nanmean(sums[:, max(i - k, 0): i + k + 1, max(j - k, 0): j + k + 1].reshape(len(sums), -1), axis=1)


def main():
    sums, lats, lons, years = annual_cell_sums()
    co = pd.read_csv(CO)
    if "lat0" not in co.columns:
        co["lat0"], co["lon0"] = co["lat"], co["lon"]
    d = pd.read_csv(ROOT / "data" / "processed" / "phase1_clean.csv")
    tr = d[d.Year <= 2009]
    rows = []
    for idx, r in co.iterrows():
        sub = tr[tr.stable_geographic_id == r.stable_geographic_id].set_index("Year")["IMD_RF"]
        yrs = [y for y in sub.index if y in set(years)]
        pos = [list(years).index(y) for y in yrs]
        i0 = int(np.abs(lats - r.lat0).argmin()); j0 = int(np.abs(lons - r.lon0).argmin())

        def score(i, j):
            x = block(sums, i, j)[pos]; y = sub.loc[yrs].values
            return np.corrcoef(x, y)[0, 1] if len(yrs) >= 8 and np.nanstd(x) > 0 else np.nan

        r0 = score(i0, j0)
        best, (bi, bj) = r0, (i0, j0)
        if len(yrs) >= 8 and (np.isnan(r0) or r0 < 0.85):
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    s = score(i0 + di, j0 + dj)
                    if not np.isnan(s) and s > best + 0.05:
                        best, (bi, bj) = s, (i0 + di, j0 + dj)
        co.loc[idx, ["lat", "lon"]] = [float(lats[bi]), float(lons[bj])]
        rows.append(dict(stable_geographic_id=r.stable_geographic_id, r_train_initial=r0, r_train_final=best,
                         moved=(bi, bj) != (i0, j0)))
    qc = pd.DataFrame(rows)
    co = co.drop(columns=[c for c in ("r_train_initial", "r_train_final", "moved") if c in co.columns]).merge(qc, on="stable_geographic_id")
    co.to_csv(CO, index=False)
    print(qc.moved.value_counts().to_dict(), "| districts with final train r < 0.7:", int((qc.r_train_final < 0.7).sum()),
          "| median r:", round(qc.r_train_final.median(), 3))


if __name__ == "__main__":
    main()
