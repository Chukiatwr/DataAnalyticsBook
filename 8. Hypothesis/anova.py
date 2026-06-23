# anova.py
#
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Generate Synthetic Data
np.random.seed(123)
data = pd.DataFrame({
    'weight_gain': np.concatenate([
        np.random.normal(20, 3, 30), 
        np.random.normal(22, 3, 30), 
        np.random.normal(28, 4, 30)  
    ]),
    'feed_type': ['Feed_A']*30 + ['Feed_B']*30 + ['Feed_C']*30
})

# 2. Assumption Check: Homogeneity of Variance
levene_stat, p_levene = stats.levene(
    data[data['feed_type'] == 'Feed_A']['weight_gain'],
    data[data['feed_type'] == 'Feed_B']['weight_gain'],
    data[data['feed_type'] == 'Feed_C']['weight_gain']
)
print(f"Levene's Test p-value: {p_levene:.4f}")

# 3. Perform One-Way ANOVA
model = ols('weight_gain ~ feed_type', data=data).fit()
anova_results = sm.stats.anova_lm(model, typ=2)

print("\n--- ANOVA Summary Table ---")
print(anova_results)

# 4. Post-hoc Analysis (Tukey HSD)
# FIX: Use .iloc[0] to avoid FutureWarning when accessing the first element of the p-value column
if anova_results['PR(>F)'].iloc[0] < 0.05:
    print("\n--- Post-hoc Test (Tukey HSD) ---")
    posthoc = pairwise_tukeyhsd(endog=data['weight_gain'], groups=data['feed_type'], alpha=0.05)
    print(posthoc)

# 5. Data Visualization
# FIX: Assign 'x' to 'hue' and set 'legend=False' to comply with Seaborn v0.14+ requirements
plt.figure(figsize=(8, 5))
sns.boxplot(
    x='feed_type', 
    y='weight_gain', 
    data=data, 
    hue='feed_type',   # Explicitly assign 'hue' to the categorical variable
    palette="viridis", 
    legend=False       # Remove legend since 'x' axis already identifies the groups
)

plt.title('Effect of Feed Type on Animal Weight Gain')
plt.xlabel('Feed Type')
plt.ylabel('Weight Gain (kg)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
