# 2-Sample t-test for Comparing Two Independent Groups
#
import numpy as np
from scipy import stats

# 1. กรณีกลุ่มตัวอย่างเป็นอิสระต่อกัน และความแปรปรวนเท่ากัน (Independent T-test: Equal Variance)
#
# ข้อมูลคะแนนสอบของนักเรียน 2 ห้อง (จำนวนคนไม่เท่ากันได้)
class_a = [78, 82, 85, 75, 70, 88, 92]
class_b = [72, 70, 75, 68, 80, 74]

# ทดสอบ Independent T-test
t_stat, p_val = stats.ttest_ind(class_a, class_b, equal_var=True)

print("--- 1. Independent T-test (Equal Variance) ---")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4f}")

if p_val < 0.05:
    print("Result: The null hypothesis is rejected (there is a significant difference between the test scores of the two classes).")
else:
    print("Result: The null hypothesis is not rejected (there is no significant difference between the test scores of the two classes).")


# 2. กรณีกลุ่มตัวอย่างเป็นอิสระต่อกัน แต่ความแปรปรวนไม่เท่ากัน (Welch's T-test: Unequal Variance)
#
# ข้อมูลรายได้พนักงาน 2 แผนก (ความแปรปรวนต่างกันชัดเจน)
dept_it = [85000, 92000, 120000, 45000, 98000]
dept_hr = [45000, 48000, 52000, 47000, 46000]

# ทดสอบ Welch's T-test
t_stat, p_val = stats.ttest_ind(dept_it, dept_hr, equal_var=False)

print("\n--- 2. Welch's T-test (Unequal Variance) ---")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4f}")
if p_val < 0.05:
    print("Result: The null hypothesis is rejected (there is a significant difference between the incomes of the two departments).")
else:
    print("Result: The null hypothesis is not rejected (there is no significant difference between the incomes of the two departments).")

# 3. กรณีกลุ่มตัวอย่างเป็นอิสระต่อกัน แต่ขนาดกลุ่มตัวอย่างเล็ก (Small Sample Size)
#
# คะแนนความพึงพอใจก่อนและหลังใช้แอปพลิเคชัน (10 คนเดิม)
before_app = [3, 4, 2, 3, 3, 2, 4, 3, 2, 3]
after_app  = [5, 4, 4, 5, 4, 3, 5, 4, 4, 5]

# ทดสอบ Paired Samples T-test
t_stat, p_val = stats.ttest_rel(before_app, after_app)

print("\n--- 3. Paired Samples T-test ---")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4f}")
if p_val < 0.05:
    print("Result: The null hypothesis is rejected (there is a significant difference in satisfaction before and after using the app).")
else:
    print("Result: The null hypothesis is not rejected (there is no significant difference in satisfaction before and after using the app).")
