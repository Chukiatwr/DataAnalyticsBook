# comparisons.py
# 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({
    "site": ["A", "B", "C", "D"],
    "co2":  [23, 45, 12, 36]
})

plt.figure(figsize=(7, 4))
plt.bar(df["site"], df["co2"])
plt.title("CO₂ Emission by Site")
plt.xlabel("Site")
plt.ylabel("CO₂ (tons)")
plt.grid(axis='y', linestyle='--', alpha=0.7) # เพิ่มเส้น Grid แนวนอน
plt.show()

# เรียงลำดับข้อมูลเพื่อให้เปรียบเทียบจากมากไปน้อย (Optional แต่แนะนำ)
df = df.sort_values("co2", ascending=True)
# ใช้ barh แทน bar เพื่อสร้างกราฟแท่งแนวนอน
# พารามิเตอร์แรกคือตำแหน่งแกน Y (Categories)
# พารามิเตอร์ที่สองคือความยาวแกน X (Values)
plt.barh(df["site"], df["co2"], color='teal')

plt.title("CO₂ Emission by Site")
plt.xlabel("CO₂ (tons)") # สลับมาเป็นแกน X
plt.ylabel("Site")       # สลับมาเป็นแกน Y

plt.grid(axis='x', linestyle='--', alpha=0.7) # เพิ่มเส้น Grid แนวตั้ง
plt.show()

df2 = pd.DataFrame({
    "site": ["A","A","B","B","C","C"],
    "year": ["2025","2026","2025","2026","2025","2026"],
    "co2":  [30, 25, 55, 48, 18, 15]
})

sns.barplot(data=df2, x="site", y="co2", hue="year")
plt.title("CO₂ Emission by Site and Year")
plt.xlabel("Site")
plt.ylabel("CO₂ (tons)")
plt.grid(axis='y', linestyle='--', alpha=0.7) # เพิ่มเส้น Grid แนวนอน
plt.show()