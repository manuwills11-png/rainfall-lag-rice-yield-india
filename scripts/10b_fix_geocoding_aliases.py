"""Resolve the districts that the geocoders could not match because the dataset uses historical / mis-spelled names.

alias -> (name to search, state to search). ``Bihar/Dhanbad`` is a pre-2000 Bihar district that now lies in Jharkhand.
Writes data/external/district_coordinates_final.csv (all 384 districts).
"""
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location("geo", Path(__file__).resolve().parent / "10_geocode_districts.py")
geo = importlib.util.module_from_spec(spec); spec.loader.exec_module(geo)

ROOT = Path(__file__).resolve().parent.parent
ALIAS = {
    "ANDHRA_PRADESH_45": ("Visakhapatnam", "Andhra Pradesh"), "ASSAM_606": ("Diphu", "Assam"),
    "BIHAR_914": ("Dhanbad", "Jharkhand"), "BIHAR_928": ("Sitamarhi", "Bihar"),
    "GUJARAT_123": ("Palanpur", "Gujarat"), "HARYANA_179": ("Bhiwani", "Haryana"),
    "HIMACHAL_PRADESH_636": ("Nahan", "Himachal Pradesh"), "JHARKHAND_938": ("Sahibganj", "Jharkhand"),
    "JHARKHAND_943": ("Jamshedpur", "Jharkhand"), "JHARKHAND_944": ("Chaibasa", "Jharkhand"),
    "KARNATAKA_80": ("Mangalore", "Karnataka"), "KARNATAKA_81": ("Karwar", "Karnataka"),
    "KERALA_651": ("Alappuzha", "Kerala"), "KERALA_662": ("Kasaragod", "Kerala"),
    "MADHYA_PRADESH_21": ("Shahdol", "Madhya Pradesh"), "MADHYA_PRADESH_36": ("Khandwa", "Madhya Pradesh"),
    "MAHARASHTRA_97": ("Alibag", "Maharashtra"), "ODISHA_708": ("Koraput", "Odisha"),
    "RAJASTHAN_155": ("Jhunjhunu", "Rajasthan"), "TAMIL_NADU_509": ("Dindigul", "Tamil Nadu"),
    "TAMIL_NADU_512": ("Madurai", "Tamil Nadu"), "TAMIL_NADU_521": ("Tiruvannamalai", "Tamil Nadu"),
    "TAMIL_NADU_83": ("Chengalpattu", "Tamil Nadu"), "TAMIL_NADU_85": ("Vellore", "Tamil Nadu"),
    "TAMIL_NADU_88": ("Tiruchirappalli", "Tamil Nadu"), "TAMIL_NADU_92": ("Tirunelveli", "Tamil Nadu"),
    "TAMIL_NADU_94": ("Nagercoil", "Tamil Nadu"), "UTTARAKHAND_241": ("Uttarkashi", "Uttarakhand"),
    "UTTAR_PRADESH_222": ("Deoria", "Uttar Pradesh"), "UTTAR_PRADESH_227": ("Rae Bareli", "Uttar Pradesh"),
    "UTTAR_PRADESH_529": ("Robertsganj", "Uttar Pradesh"), "UTTAR_PRADESH_532": ("Naugarh", "Uttar Pradesh"),
    "WEST_BENGAL_816": ("Cooch Behar", "West Bengal"), "WEST_BENGAL_823": ("Tamluk", "West Bengal"),
    "WEST_BENGAL_824": ("Midnapore", "West Bengal"),
}

c = pd.read_csv(ROOT / "data" / "external" / "district_coordinates.csv")
for i, r in c[c.source == "FAILED"].iterrows():
    name, state = ALIAS[r.stable_geographic_id]
    res = geo.openmeteo(name, state)
    if res is None:
        res = geo.nominatim(name, state)
    if res is None:
        print("STILL FAILED", r.stable_geographic_id, name, state); continue
    c.loc[i, ["lat", "lon", "source", "match"]] = [res[0], res[1], "alias:" + res[2], res[3]]
c.to_csv(ROOT / "data" / "external" / "district_coordinates_final.csv", index=False)
print(c.source.value_counts())
