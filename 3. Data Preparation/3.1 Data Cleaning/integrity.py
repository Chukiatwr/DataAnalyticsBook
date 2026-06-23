# integrity.py

import pandas as pd
import numpy as np

df = pd.DataFrame({
    "id": ["C01","C02","C03","C04","C05","C06","C07"],
    "age": [28, 41, 15, 220, 33, np.nan, 29],                      # 220 is invalid
    "gender": ["Male","Female","Other","Unknown","Female","Female","Male"], # invalid "Unknown"
    "height": [170, 158, 165, 172, 1000, 160, 175],                # 1000 is outlier
    "temp": [36.7, np.nan, 98.6, 37.8, 99.5, 36.3, 37.0],          # Some is Fahrenheit
    "temp_unit": ["C","C","F","C","F","C","C"],                    # Mixed C/F
    "amount": [2500, 0, 120, -50, 9999999, 350, 700],              # Invalid 0/-50/9,999,999
})
print("=========== Original Data =============")
print(df)

# 1. Outlier Treatment

# Capped with business-defined boundary e.g. 120-220 cm for height
df["height_capped"] = df["height"].clip(lower=120, upper=220)

# Capped with IQR
def cap_with_iqr(data, multiplier=1.5):
    # Calculate Q1, Q3, และ IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    # Cap the data with bounds
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    capped_data = np.clip(data, lower_bound, upper_bound)
    
    return capped_data, lower_bound, upper_bound

df["height_iqr"], lower, upper = cap_with_iqr(df["height"].astype(float), multiplier=1.5)
print("\n=========== Cap with Boundary and IQR =============")
print(f"Height capped with IQR between {lower} and {upper}")
print(df[["height", "height_capped", "height_iqr"]])


# 2.1-3 Data Validation

# 2.1 Age range
df["invalid_age"] = ~df["age"].between(0, 120, inclusive="both")

# 2.2 Gender
allowed_gender = {"Male","Female","Other"}
df["invalid_gender"] = ~df["gender"].isin(allowed_gender)

# 2.3 Amount range
BUSINESS_MAX = 500_000
df["invalid_amount"] = (df["amount"] <= 0) | (df["amount"] > BUSINESS_MAX)

print("\n=========== Invalid Data Flags =============")
print(df[["age", "invalid_age", "gender", "invalid_gender", "amount", "invalid_amount"]])


# 3 Change temperature unit to Celsius

def to_celsius(temp, unit):
    if pd.isna(temp): 
        return np.nan
    if unit == "C":
        return temp
    if unit == "F":
        return (temp - 32) * 5/9
    return np.nan  # unknown unit

df["temp_c"] = [to_celsius(t, u) for t, u in zip(df["temp"], df["temp_unit"])]
df["invalid_temp_c"] = ~pd.Series(df["temp_c"]).between(30, 45, inclusive="both")  # 30-45 C

print("\n=========== Temperature Unit =============")
print(df[["id", "temp","temp_unit","temp_c","invalid_temp_c"]])