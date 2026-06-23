# violin.py
#
import matplotlib.pyplot as plt
import numpy as np

# ข้อมูลตัวอย่าง: การปล่อยก๊าซ (tCO2e) สำหรับภูมิภาคต่างๆ
np.random.seed(123)
regions = ["EMEA", "APAC"]
emissions = [
    np.random.normal(600, 150, 30),   # EMEA
    np.random.normal(450, 60, 30)    # APAC
]

plt.figure(figsize=(7, 4))

# สร้าง Violin Plot
# showmedians=True เพื่อแสดงเส้นค่ากลาง (คล้ายเส้นสีแดงใน Box Plot เดิม)
parts = plt.violinplot(emissions, showmedians=True, showextrema=True)

# ปรับแต่งสีของตัว Violin (Bodies)
for pc in parts['bodies']:
    pc.set_facecolor('skyblue')    # สีฟ้าเหมือนเดิม
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)              # ปรับความโปร่งใสเล็กน้อยให้ดูสวยงาม

# ปรับแต่งสีของเส้นค่ากลาง (Median line)
parts['cmedians'].set_color('red')
parts['cmedians'].set_linewidth(2)

# กำหนดชื่อแกน X (เนื่องจาก violinplot ของ matplotlib พื้นฐานไม่รับ labels โดยตรง)
plt.xticks(np.arange(1, len(regions) + 1), labels=regions)

plt.title("Site-Level Emissions by Region (tCO2e)", fontsize=16)
plt.ylabel("Emissions per site (tCO2e)")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()