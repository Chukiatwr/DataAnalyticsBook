# heatmap.py - Correlation Heatmap for ESG Variables
# 
# Correlation Heatmap Demo (ESG_Score depends on other variables)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(123)  # Set random seed for reproducibility
# -------------------------------------------
# Create independent ESG-related variables
# -------------------------------------------
n = 80
carbon_emission = np.random.lognormal(mean=np.log(100), sigma=0.5, size=n)         # Higher means worse ESG
employee_satisfaction = np.random.normal(7, 1.2, n)                               # Higher means better ESG
governance_score = np.random.normal(75, 5, n)                                     # Higher means better ESG

# -------------------------------------------
# Create ESG_Score as a combination of other variables
# We assume:
# - High carbon emission → lowers ESG score
# - High employee satisfaction and governance → increase ESG score
# - Random noise adds realism
ESG_Score = (
    85
    - 0.15 * carbon_emission       # inverse effect
    + 4.0 * employee_satisfaction  # positive effect
    + 1 * governance_score         # positive effect
    + np.random.normal(0, 5, n)    # noise
)

# Combine into a DataFrame
df = pd.DataFrame({
    'ESG_Score': ESG_Score,
    'Carbon_Emission': carbon_emission,
    'Employee_Satisfaction': employee_satisfaction,
    'Governance_Score': governance_score
})

# -------------------------------------------
# Compute correlation matrix
corr_matrix = df.corr(method='pearson')

# -------------------------------------------
# Plot Heatmap
plt.figure(figsize=(14, 5))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Correlation Heatmap of ESG Variables", fontsize=16)
plt.show()