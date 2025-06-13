import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Dataset after feature engineering (before clustering)
df = pd.read_csv("data/feature-engineered/movies_normalized.csv") 

# Clustering: k-means
# A technique for classifying movie data by type and finding hidden patterns
cluster_features = ['budget', 'weighted_score']
X = df[cluster_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# PCA (2D reduction)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# K=3 clustering
kmeans3 = KMeans(n_clusters=3, random_state=42)
labels3 = kmeans3.fit_predict(X_scaled)

# # Visualization
# plt.figure(figsize=(12, 5))
# # K=3
# plt.subplot(1, 2, 1)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels3, cmap='viridis')
# plt.title("K-Means Clustering (k=3)")
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.show() 

# k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Calculate average by cluster
cluster_summary = df.groupby('cluster')[['budget', 'weighted_score']].mean().round(2)
cluster_summary['count'] = df['cluster'].value_counts().sort_index()


# Naming clusters
# Manually name clusters by looking at their characteristics

# Checking the quantile
print(df['budget'].describe())

# Calculate average is_hit value separately for each cluster
hit_rate_per_cluster = df.groupby('cluster')['is_hit'].mean()

cluster_names = {}
for cluster in cluster_summary.index:
    row = cluster_summary.loc[cluster]
    budget = row['budget']
    hit_rate = hit_rate_per_cluster[cluster]  # 👈 여기만 따로 불러옴

    if budget >= 0.78:
        name = "high-budget-success" if hit_rate >= 0.5 else "high-budget-failure"
    elif budget < 0.70:
        name = "low-budget-success" if hit_rate >= 0.5 else "low-budget-failure"
    else:
        name = "mid-budget"

    cluster_names[cluster] = name


# Add name mapping
df['cluster_label'] = df['cluster'].map(cluster_names)

# Print Result
print("Average Summary by Cluster:")
print(cluster_summary)
print("\nMapping Cluster Names:")
for k, v in cluster_names.items():
    print(f"Cluster {k}: {v}")


df.to_csv("data/feature-engineered/movies_clustered.csv", index=False)

print("Saved Dataset: data/feature-engineered/movies_clustered.csv")