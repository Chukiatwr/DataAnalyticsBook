# consistency.py

import pandas as pd
import numpy as np
import re

# 1) Synthesize raw inconsistent data
raw = pd.DataFrame({
    "name": ["  anna  ", "BOB", "cindy", "donald  "],
    "city": ["bkk", "Bangkok", "BANGKOK ", "Krung Thep"],
    "gender": ["female", "M", "FEMALE", "m"]
})
print("========= Raw Data =========")
print(raw)

# Name: remove extra spaces, title case
def std_person_name(x: str):
    if x is None:
        return np.nan
    return re.sub(r"\s+", " ", str(x).strip()).title()

# Gender: normalize
def std_gender(x: str):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().lower()
    if s in {"m", "male"}:
        return "Male"
    if s in {"f", "female"}:
        return "Female"
    return np.nan  # Don't recognize

# City normalizer (lower -> map -> title case)
CITY_MAP = {
    "bkk": "Bangkok",
    "bangkok": "Bangkok",
    "krung thep": "Bangkok",
}
def std_city(x: str):
    if x is None:
        return np.nan
    s = str(x).strip().lower()
    v = CITY_MAP.get(s, s)
    return v.strip().title()

# 2) Clean and standardize data
df = raw.copy()
df["name_std"]   = df["name"].map(std_person_name)
df["gender_std"] = df["gender"].map(std_gender)
df["city_std"]   = df["city"].map(std_city)

# 3) Report standardized data
print("\n========= Standardized Data =========")
print(df[[
    "name","name_std",
    "gender","gender_std",
    "city","city_std"
]])