# aggregate.py

import pandas as pd
import numpy as np

pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 85)

# ---------------------------------------------
# 1) Create a synthetic ESG dataset
# ---------------------------------------------
np.random.seed(42)
companies = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
sectors = ["Energy", "Finance", "Industrials"]
regions = ["EMEA", "APAC"]

rows = []
for c in companies:
    sector = np.random.choice(sectors)
    region = np.random.choice(regions)
    # Generate monthly observations for 2026
    for dt in pd.date_range("2026-01-01", "2026-12-01", freq="MS"):
        e = np.clip(np.random.normal(65, 12), 0, 100)  # E score 0..100
        s = np.clip(np.random.normal(60, 10), 0, 100)  # S score
        g = np.clip(np.random.normal(62, 11), 0, 100)  # G score
        emissions = max(0, np.random.lognormal(mean=10, sigma=0.7))  # tCO2e
        water = max(0, np.random.lognormal(mean=6, sigma=0.6))       # m^3
        rows.append([dt, c, sector, region, e, s, g, emissions, water]) # , controversies])

df = pd.DataFrame(rows, columns=[
    "date","company","sector","region", "E_score","S_score","G_score",
    "emissions_tCO2e","water_m3" # ,"controversies"
])

# Optional: introduce some missingness to show robust aggregation
mask = np.random.rand(len(df)) < 0.03
df.loc[mask, "E_score"] = np.nan

# ---------------------------------------------
# 2) Sector-level mean ESG scores (simple average)
# ---------------------------------------------
sector_mean = (
    df.groupby("sector", as_index=False)[["E_score","S_score","G_score"]]
      .mean(numeric_only=True)
)

# ---------------------------------------------
# 3) Multi-metric aggregation per sector (count, median, sum, quantile)
# ---------------------------------------------
sector_agg = (
    df.groupby("sector").agg(
        companies=("company","nunique"),
        E_median=("E_score","median"),
        emissions_sum=("emissions_tCO2e","sum"),
        emissions_p90=("emissions_tCO2e", lambda s: s.quantile(0.90)),
    ).reset_index()
)

# ---------------------------------------------
# 4) Region x Sector pivot table
#    - Mean E score, Sum of emissions
# ---------------------------------------------
pivot_rs = pd.pivot_table(
    df,
    index="region",
    columns="sector",
    values=["E_score","emissions_tCO2e"],
    aggfunc={"E_score":"mean","emissions_tCO2e":"sum"},
)

# ---------------------------------------------
# 5) Quick prints (or inspect in a notebook)
# ---------------------------------------------
print("\n=== Sector simple means ===\n", sector_mean)
print("\n=== Sector multi-metric aggregation ===\n", sector_agg)
print("\n=== Region x Sector pivot (E_score mean, Emissions sum) ===\n", pivot_rs)