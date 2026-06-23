# concat.py

import pandas as pd
import numpy as np

# ESG data of companies in 2023
esg_2023 = pd.DataFrame({
    "company": ["A", "B", "C"],
    "year": [2023, 2023, 2023],
    "carbon_emission": [1200, 950, 1430],
    "water_usage": [300, 250, 410]
})

# ESG data of companies in 2024
esg_2024 = pd.DataFrame({
    "company": ["A", "B", "C"],
    "year": [2024, 2024, 2024],
    "carbon_emission": [1100, 900, 1380],
    "water_usage": [280, 240, 390]
})

# Display the original data
print("--- ESG Data for 2023 ---")
print(esg_2023)
print("\n--- ESG Data for 2024 ---")
print(esg_2024)

print("\n--- Concatenating vertically (axis=0) ---")
esg_all_years = pd.concat(
    [esg_2023, esg_2024],
    axis=0,
    ignore_index=True
)
print(esg_all_years)

environment_data = pd.DataFrame({
    "company": ["A", "B", "C"],
    "water_usage": [280, 240, 390]
})

social_governance_data = pd.DataFrame({
    "emp_sat": [85, 88, 79],
    "anti_crpt": ["Yes", "Yes", "No"]
})

# Display the original data
print("\n--- Environment Data ---")
print(environment_data)
print("\n--- Social & Governance Data ---")
print(social_governance_data)

print("\n--- Concatenating horizontally (axis=1) ---")
esg_combined = pd.concat(
    [environment_data, social_governance_data],
    axis=1
)

print(esg_combined)