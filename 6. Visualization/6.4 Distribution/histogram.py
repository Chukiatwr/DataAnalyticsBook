# histogram.py
#
import matplotlib.pyplot as plt
import pandas as pd

# ข้อมูลสมมติ: อายุของลูกค้า 20 คน
ages = [22, 25, 25, 30, 32, 35, 38, 40, 42, 45, 50, 55, 28, 24, 31, 33, 37, 28, 29, 34] 

plt.figure(figsize=(8, 5))
plt.hist(ages, bins=10, color='skyblue', edgecolor='grey')
plt.title('Distribution of Customer Age', fontsize=16)
plt.xlabel('Age')
plt.ylabel('Number of Customers')
plt.show()