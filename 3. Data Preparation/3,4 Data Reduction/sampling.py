# sampling.py

import pandas as pd
import numpy as np

# pandas settings for better display of dataframes
pd.set_option('display.float_format', '{:.4f}'.format)

# Synthetic data of ESG ratings
esg_data = pd.DataFrame({
    'company_id': range(10000),
    'esg_rating': np.random.choice(
        ['High', 'Medium', 'Low'], 
        10000, 
        p=[0.15, 0.60, 0.25]  # Imbalanced
    )
})

print("Original distribution:")
print(esg_data['esg_rating'].value_counts(normalize=True))

# Simple random sampling
sample = esg_data.sample(frac=0.25, random_state=11)  # 25% sample
print("\nSample distribution from simple sampling:")
print(sample['esg_rating'].value_counts(normalize=True))

# Stratified sampling
# Split: 80% train, 20% test (stratified by 'esg_rating')
from sklearn.model_selection import train_test_split
train, test = train_test_split(
    esg_data,
    test_size=0.25,
    stratify=esg_data['esg_rating'],  # Critical: column to stratify by
    random_state=11
)

print("\nTest set distribution from stratified sampling:")
print(test['esg_rating'].value_counts(normalize=True))