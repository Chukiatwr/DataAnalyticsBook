# boxplot.py
#
import matplotlib.pyplot as plt
import numpy as np

# Example data: emissions (tCO2e) for different regions
np.random.seed(123)
regions = ["EMEA", "APAC"]
emissions = [
    np.random.normal(600, 150, 30),   # EMEA
    np.random.normal(450, 60, 30)     # APAC
]

plt.figure(figsize=(7, 4))
plt.boxplot(emissions, labels=regions, patch_artist=True, widths=0.4,
            boxprops=dict(facecolor="skyblue", color="black"),
            medianprops=dict(color="red", linewidth=2))

plt.title("Site-Level Emissions by Region (tCO2e)", fontsize=16)
plt.ylabel("Emissions per site (tCO2e)")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()