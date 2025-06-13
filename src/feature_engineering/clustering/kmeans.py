import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Save path
save_dir = "results/feature-engineering/clustering"
os.makedirs(save_dir, exist_ok=True)

# Dataset with feature engineering; before clustering
df = pd.read_csv("data/feature-engineered/movies_normalized.csv") 

# Clustering: k-means
# 영화 데이터를 유형별로 분류해서 숨겨진 패턴을 찾아내기 위한 기법

# 1. Specify features to use
cluster_features = ['budget', 'weighted_score']

X = df[cluster_features]

# StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method - to determine the most appropriate value for k
inertia = []
K_range = range(1, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, 'bo-')

plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig(os.path.join(save_dir, "elbow.png"))
plt.show()

# PCA 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 1. K=3 clustering
kmeans3 = KMeans(n_clusters=3, random_state=42)
labels3 = kmeans3.fit_predict(X_scaled)

# 2. K=4 clustering
kmeans4 = KMeans(n_clusters=4, random_state=42)
labels4 = kmeans4.fit_predict(X_scaled)

# Visualization
plt.figure(figsize=(12, 5))
# K=3
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels3, cmap='viridis')
plt.title("K-Means Clustering (k=3)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
# K=4
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels4, cmap='viridis')
plt.title("K-Means Clustering (k=4)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "kmeans_comparison_k3_k4.png"))
plt.show() 

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}, Silhouette Score={score:.4f}")

# Save results
with open(os.path.join(save_dir, "silhouette_scores.txt"), "w") as f:
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        f.write(f"k={k}, Silhouette Score={score:.4f}\n")
        
# K=3