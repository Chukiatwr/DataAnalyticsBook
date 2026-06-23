import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. สร้างตัวอย่างข้อมูล (Simulation Data)
data = {
    'Age': [25, 30, np.nan, 45, 22],
    'Salary': [50000, 54000, 60000, np.nan, 48000],
    'City': ['BKK', 'CNX', 'BKK', 'HKT', np.nan],
    'Bought': ['No', 'Yes', 'No', 'Yes', 'No']
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
print("\nData Types:\n", df.dtypes)


# 2. แยก Features และ Target
X = df.drop('Bought', axis=1)
y = df['Bought']

# 3. กำหนดกลุ่มตัวแปร (Numerical vs Categorical)
num_features = ['Age', 'Salary']
cat_features = ['City']

# 4. สร้าง Pipeline สำหรับจัดการข้อมูลแต่ละประเภท
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # เติมค่าว่างด้วยมัธยฐาน
    ('scaler', StandardScaler())                  # ปรับมาตรฐานข้อมูล (Z-score)
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # เติมค่าว่างด้วยฐานนิยม
    ('onehot', OneHotEncoder(handle_unknown='ignore'))   # แปลงเป็น 0, 1
])

# 5. รวมการทำงานเข้าด้วยกัน (ColumnTransformer)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# 6. รันกระบวนการ Preprocessing
X_processed = preprocessor.fit_transform(X)

print("Processed Data Shape:", X_processed.shape)
print("Example of processed row:\n", X_processed[0])