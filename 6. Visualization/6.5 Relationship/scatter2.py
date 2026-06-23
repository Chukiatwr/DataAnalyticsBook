# scatter2.py
#
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)
x = np.random.rand(50) * 10
y = x * 2 + np.random.randn(50) * 2 + 10

plt.figure(figsize=(6, 4))
plt.scatter(x, y, color='green', alpha=0.7, edgecolors='k')
plt.title("Positive Correlation: Study Hours vs. Exam Scores", fontsize=14)
plt.xlabel("Study Hours")
plt.ylabel("Exam Scores")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

np.random.seed(123)
x = np.random.rand(50) * 15 + 15
y = -x * 1.5 + np.random.randn(50) * 5 + 100

plt.figure(figsize=(6, 4))
plt.scatter(x, y, color='red', alpha=0.7, edgecolors='k')
plt.title("Negative Correlation: Temperature vs. Sweater Sales", fontsize=14)
plt.xlabel("Temperature (°C)")
plt.ylabel("Sweater Sales")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

np.random.seed(123)
x = np.random.rand(50) * 10000
y = np.random.rand(50) * 10000 - 1000

plt.figure(figsize=(6, 4))
plt.scatter(x, y, color='blue', alpha=0.7, edgecolors='k')
plt.title("No Correlation: Daily Steps vs. Profit from Stock", fontsize=14)
plt.xlabel("Daily Steps")
plt.ylabel("Profit from Stock (Baht)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ข้อมูลแบบ Quadratic: ปริมาณปุ๋ย (x) กับ ผลผลิตพืช (y) 
# ใส่มากไปผลผลิตก็ลดลงเนื่องจากดินเป็นพิษ
x = np.linspace(0, 10, 50)
y = -2 * (x - 5)**2 + 50 + np.random.normal(0, 5, 50)

plt.figure(figsize=(6, 4))
plt.scatter(x, y, color='purple', alpha=0.7, edgecolors='k')

plt.title("Non-linear Correlation: Fertilizer Amount vs. Crop Yield", fontsize=14)
plt.xlabel("Fertilizer Amount")
plt.ylabel("Crop Yield")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()