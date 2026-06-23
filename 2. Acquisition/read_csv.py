# read_csv.py

import pandas as pd

# อ่านไฟล์ CSV
df = pd.read_csv("data.csv")
print(df.head())

# sep กำหนดตัวคั่นเป็นเครื่องหมาย ;
df = pd.read_csv("data.csv", sep=';')

# header ใช้บรรทัดที่ 2 (index = 1) เป็น header
df = pd.read_csv("data.csv", header=1)

# skiprows ข้าม 2 บรรทัดแรก
df = pd.read_csv("data.csv", skiprows=2)

# usecols เลือกอ่านเฉพาะบางคอลัมน์
df = pd.read_csv("data.csv", usecols=["ID", "Name"])

# names กำหนดชื่อคอลัมน์เอง
df = pd.read_csv("data.csv", header=None, names=["ID", "Name", "Age"])

# encoding กำหนดการเข้ารหัส (ภาษา) ของไฟล์
df = pd.read_csv("data.csv", encoding="utf-8")  
df = pd.read_csv("data.csv", encoding="tis-620")  # สำหรับภาษาไทยบางไฟล์

# แสดงตัวอย่าง 5 แถวแรก
print(df.head())
