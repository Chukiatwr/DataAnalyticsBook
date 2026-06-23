# 1-Sample t-test for Quality Control of Beverage Can Filling
#
import numpy as np
from scipy import stats

# --- 1. Define Standard and Simulation Parameters ---
# The standard volume to test against (mu_0)
target_mean = 330.0

# Parameters for generating simulated sample data
desired_sample_mean = 329.7  # Simulate the machine underfilling slightly
desired_sample_std = 0.8     # Assumed variability (standard deviation)
sample_size = 30             # Number of cans sampled

# --- 2. Generate Simulated Sample Data with Seed 123 ---
np.random.seed(123) 
data = np.random.normal(loc=desired_sample_mean, scale=desired_sample_std, 
                        size=sample_size)

# 3. Calculate Sample Statistics
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1) # ddof=1 for Sample Standard Deviation (s)
alpha = 0.05 # Significance level (5%)

# 4. Perform the One-Sample T-Test
# ttest_1samp calculates the T-statistic and P-value.
t_statistic, p_value = stats.ttest_1samp(a=data, popmean=target_mean, 
                                         alternative='two-sided')

# 5. Display Results
print(f"--- One-Sample T-Test Results (Quality Control) ---")
print(f"Target Volume (mu_0): {target_mean} ml")
print("-" * 50)
print(f"Calculated Sample Mean (x-bar): {sample_mean:.4f} ml")
print(f"Sample Standard Deviation (s): {sample_std:.4f} ml")
print(f"Sample Size (n): {sample_size}")
print("-" * 50)
print(f"T-Statistic: {t_statistic:.4f}")
print(f"P-Value: {p_value:.4f}")
print(f"Significance Level (alpha): {alpha}")
print("-" * 50)

# 6. Statistical Decision and Business Conclusion
if p_value < alpha:
    print(f"Decision: **REJECT H0** (P-Value < Alpha).")
    print(f"Conclusion: **Significantly differs** from {target_mean} ml.")
    if sample_mean < target_mean:
        print(f"Action: **Underfilling problem**; recalibration is required.")
    else:
        print(f"Action: **Overfilling problem** (cost issue); recalibration is required.")
else:
    print(f"Decision: **FAIL TO REJECT H0** (P-Value >= Alpha).")
    print(f"Conclusion: NOT significantly differ from {target_mean} ml.")
    print(f"Action: The machine is operating within statistical tolerance.")
