# bubble.py
#
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# การจัดเตรียมข้อมูลเชิงประยุกต์ทางการตลาด
data = {
    'Marketing_Budget': [12, 25, 35, 45, 55, 68, 75, 88, 95, 110],
    'Sales_Revenue': [20, 38, 42, 55, 62, 70, 82, 85, 98, 105],
    'Customer_Reach': [100, 250, 300, 450, 500, 750, 800, 950, 1100, 1300]
}
df = pd.DataFrame(data)

# การสร้างแผนภูมิเชิงความสัมพันธ์แบบหลายมิติ
fig, ax = plt.subplots(figsize=(7, 5))

scatter = ax.scatter(
    df['Marketing_Budget'], 
    df['Sales_Revenue'], 
    s=df['Customer_Reach'],      # มิติที่ 3: ขนาดวงกลมตามการเข้าถึงลูกค้า
    c=df['Sales_Revenue'],       # มิติที่ 4: ไล่เฉดสีตามรายได้
    cmap='viridis',              # ชุดสีที่เหมาะสมสำหรับการอ่านค่า
    alpha=0.6,                   # กำหนดความโปร่งแสงเพื่อลดการซ้อนทับ
    edgecolors="white",          # เส้นขอบเพื่อแยกแยะแต่ละจุดข้อมูล
    linewidth=1.5
)

# การกำหนดองค์ประกอบเชิงบรรยาย
ax.set_title('Correlation Analysis: Budget, Revenue, and Reach', fontsize=16, pad=20)
ax.set_xlabel('Marketing Budget (Thousand Units)', fontsize=12)
ax.set_ylabel('Sales Revenue (Thousand Units)', fontsize=12)

# การเพิ่มแถบอธิบายสีและตารางพื้นหลัง
cbar = plt.colorbar(scatter)
cbar.set_label('Revenue Intensity')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()