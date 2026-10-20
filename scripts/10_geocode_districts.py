"""Geocode every district in the dataset to a representative lat/lon (Open-Meteo geocoder = district HQ town; OSM Nominatim fallback).

Output: data/external/district_coordinates.csv  (resumable; cached rows are skipped)
Run from project root:  python scripts/10_geocode_districts.py
"""
import sys, time, unicodedata
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "external" / "district_coordinates.csv"
UA = {"User-Agent": "rainfall-lag-crop-yield-research/1.0 (academic project)"}

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("-", " ").split())

def nominatim(district, state):
    for q in (f"{district} district, {state}, India", f"{district}, {state}, India"):
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": q, "format": "json", "limit": 5, "addressdetails": 0}, headers=UA, timeout=30)
        time.sleep(1.1)
        for h in r.json():
            if norm(state) in norm(h["display_name"]):
                return float(h["lat"]), float(h["lon"]), "nominatim", h["display_name"]
    return None

def openmeteo(district, state):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": district, "count": 20, "country_code": "IN", "language": "en"}, timeout=30).json()
    for h in r.get("results", []):
        if norm(h.get("admin1", "")) == norm(state):
            return h["latitude"], h["longitude"], "open-meteo", f"{h['name']}, {h.get('admin2')}, {h.get('admin1')}"
    return None

def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "phase1_clean.csv")
    d = (df.groupby(["stable_geographic_id", "State_Clean", "District_Clean"]).size().reset_index(name="n"))
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["stable_geographic_id"])
    have = set(done["stable_geographic_id"])
    rows = done.to_dict("records")
    for i, r in d.iterrows():
        if r.stable_geographic_id in have:
            continue
        # dataset spells some states in upper case; geocoders want proper names
        state = r.State_Clean
        res = None
        try:
            res = openmeteo(r.District_Clean, state) or nominatim(r.District_Clean, state)
        except Exception as e:
            print("ERR", r.District_Clean, e, file=sys.stderr)
        lat, lon, src, note = res if res else (None, None, "FAILED", "")
        rows.append(dict(stable_geographic_id=r.stable_geographic_id, State=state, District=r.District_Clean,
                         lat=lat, lon=lon, source=src, match=note))
        print(f"{i+1}/{len(d)} {state}/{r.District_Clean} -> {src} {lat} {lon}", flush=True)
        if len(rows) % 20 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(rows).to_csv(OUT, index=False)

if __name__ == "__main__":
    main()
