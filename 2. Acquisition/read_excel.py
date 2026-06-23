# read_excel.py

import pandas as pd

# อ่านชีตแรกของไฟล์
df = pd.read_excel("sales.xlsx")

# อ่านชีตตามชื่อ
df = pd.read_excel("sales.xlsx", sheet_name="2025_Q1")

# อ่านชีตตามลำดับ index (0 = ชีตแรก)
df = pd.read_excel("sales.xlsx", sheet_name=0)

# อ่านหลายชีตพร้อมกัน (ได้ dict ของ DataFrames)
dfs = pd.read_excel("sales.xlsx", sheet_name=["2025_Q1", "2025_Q2"])

# อ่านเฉพาะคอลัมน์ A และ C
df = pd.read_excel("sales.xlsx", usecols=["A", "C"])

# อ่านเฉพาะคอลัมน์ตามชื่อ
df = pd.read_excel("sales.xlsx", usecols=["Product", "Revenue"])

# ข้ามบรรทัดแรก (เช่น metadata หรือคำอธิบาย)
df = pd.read_excel("sales.xlsx", skiprows=1)

# ข้ามหลายบรรทัด (เช่น 0, 2, 3)
df = pd.read_excel("sales.xlsx", skiprows=[0, 2, 3])

# อ่านเฉพาะ 10 แถวแรก
df = pd.read_excel("sales.xlsx", nrows=10)

# ใช้แถวที่ 2 (index=1) เป็น header
df = pd.read_excel("sales.xlsx", header=1)

