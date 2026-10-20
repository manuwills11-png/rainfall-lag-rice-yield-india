"""Download IMD 0.25-degree daily gridded rainfall (1990-2017) and 1-degree daily Tmax with ``imdlib``.

IMD (India Meteorological Department, Pune) is the same agency that produced the ``IMD_RF`` / ``IMD_Tmax`` columns
of the yield dataset, so this is the most consistent time-resolved source for the rainfall-lag hypothesis.
Files are stored as yearly binary .GRD under data/external/imd_grid/ (git-ignored, ~25 MB/year).
"""
from pathlib import Path
import imdlib as imd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "external" / "imd_grid"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for var in ("rain", "tmax"):
        for yr in range(1990, 2018):
            try:
                imd.get_data(var, yr, yr, fn_format="yearwise", file_dir=str(OUT))
            except Exception as e:           # keep going; missing years reported at the end
                print("FAILED", var, yr, repr(e)[:200], flush=True)
