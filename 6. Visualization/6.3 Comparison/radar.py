# radar.py
#
import numpy as np
import matplotlib.pyplot as plt

# 1. กำหนดหัวข้อการเปรียบเทียบ (Attributes)
categories = ['Battery', 'Camera', 'Design', 'Performance', 'Display', 'Price Value']
num_vars = len(categories)

# 2. กำหนดข้อมูลคะแนน (Scale 1-10)
# ข้อมูลต้องปิดท้ายด้วยค่าแรกเพื่อให้เส้นเชื่อมกันเป็นรูปทรงปิด
values_a = [8, 9, 7, 9, 8, 6]
values_a += values_a[:1]

values_b = [9, 7, 8, 7, 9, 9]
values_b += values_b[:1]

# 3. คำนวณมุมของแต่ละแกนบนวงกลม (360 องศา หารตามจำนวนตัวแปร)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # ปิดวงกลม

# 4. สร้างกราฟ
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# วาดโครงสร้างและระบายสีพื้นหลังของแต่ละรุ่น
# Model A
ax.plot(angles, values_a, color='#1f77b4', linewidth=2, label='Model A (Flagship)')
ax.fill(angles, values_a, color='#1f77b4', alpha=0.25)

# Model B
ax.plot(angles, values_b, color='#ff7f0e', linewidth=2, label='Model B (Mid-Range)')
ax.fill(angles, values_b, color='#ff7f0e', alpha=0.25)

# 5. ตกแต่งแกนและป้ายกำกับ
ax.set_theta_offset(np.pi / 2) # ให้จุดแรกเริ่มที่ตำแหน่ง 12 นาฬิกา
ax.set_theta_direction(-1)     # หมุนตามเข็มนาฬิกา

# ตั้งชื่อแกนแต่ละด้าน
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)

# ตั้งค่าระดับคะแนน (Grid lines)
ax.set_rlabel_position(0)
plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=8)
plt.ylim(0, 10)

plt.title('Smartphone Features Comparison', size=18)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

plt.show()