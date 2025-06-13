import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Dataset with basic feature engineering
df2 = pd.read_csv("data/feature-engineered/movies_feature_engineered_basic.csv")
df_movies_normalized = df2.copy()

# Log scale columns
log_cols = ['budget', 'gross', 'votes']
for col in log_cols:
    df_movies_normalized[col] = np.log1p(df_movies_normalized[col])  

# Normalization of target column
minmax_cols = ['score', 'runtime']
scaler_minmax = MinMaxScaler()
df_movies_normalized[minmax_cols] = scaler_minmax.fit_transform(df_movies_normalized[minmax_cols])

# Normalize log-transformed columns
scaler_log = MinMaxScaler()
df_movies_normalized[log_cols] = scaler_log.fit_transform(df_movies_normalized[log_cols])

# Backup
df_movies_normalized.to_csv("data/feature-engineered/movies_normalized.csv", index=False)

fig, axes = plt.subplots(len(log_cols + minmax_cols), 2, figsize=(12, 5 * len(log_cols + minmax_cols)))

for i, col in enumerate(log_cols + minmax_cols):
    # Original Distribution
    sns.histplot(df2[col], bins=50, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f"Original Distribution of {col}")
    
    # After normalization
    sns.histplot(df_movies_normalized[col], bins=50, kde=True, ax=axes[i, 1], color='green')
    axes[i, 1].set_title(f"Normalized Distribution of {col}")

plt.tight_layout()
plt.show()

print("Saved Dataset: data/feature-engineered/movies_normalized.csv")