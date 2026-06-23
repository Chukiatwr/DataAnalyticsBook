# feature_creation.py
#
import pandas as pd
import numpy as np

# Simulated credit card spending data
df = pd.DataFrame({
    'customer_id': [101, 101, 102],
    'transaction_date': pd.to_datetime(['2026-01-01', '2026-01-12', '2026-02-05']),
    'spend_amount': [5000, 2000, 15000],
    'credit_limit': [20000, 20000, 30000]
})

# 1. Aggregation (spend_amount grouped by customer_id)
agg_features = df.groupby('customer_id')['spend_amount'].sum().rename('total_spend')

# 2. Temporal (transaction_month)
df['tran_month'] = df['transaction_date'].dt.month

# 3. Ratios (spend amount per credit limit)
df['utilization_ratio'] = df['spend_amount'] / df['credit_limit']

# 4. Binning (of spend_amount into categories)
df['spend_level'] = pd.cut(df['spend_amount'], bins=[0, 3000, 10000, np.inf], labels=['Low', 'Medium', 'High'])

print(df[['customer_id', 'tran_month', 'utilization_ratio', 'spend_level']])