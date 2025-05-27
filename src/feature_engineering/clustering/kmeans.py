import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# feature engineering된 데이터셋 (clustering 전)
df = pd.read_csv("data/feature-engineered/movies_normalized.csv") 

# Clustering: k-means
# 영화 데이터를 유형별로 분류해서 숨겨진 패턴을 찾아내기 위한 기법
# 1. 사용할 feature 지정 (One-Hot Encoding이 필요한 컬럼 포함)
cluster_features = ['budget', 'weighted_score']

# # 2. One-Hot Encoding (drop_first=True로 차원 줄이기)
# df_encoded = pd.get_dummies(df, columns=['genre_top10'], drop_first=True)

# # 3. 인코딩된 컬럼명 추출
# encoded_columns = [col for col in df_encoded.columns if col.startswith('genre_top10_')]

# 4. 최종 feature 조합
X = df[cluster_features]

# 5. 정규화 (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method - k에 어떤 값이 들어가야 가장 적절한지 알 수 있음
# inertia = []
# K_range = range(1, 10)
# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X_scaled)
#     inertia.append(kmeans.inertia_)

# plt.plot(K_range, inertia, 'bo-')

# plt.xlabel("k")
# plt.ylabel("Inertia")
# plt.title("Elbow Method")
# # plt.legend()
# plt.show()

# # k=3, k=4인 경우 inertia가 급격히 떨어지는 구간; 점선으로 표시 후 다시 나타냄
# plt.plot(K_range, inertia, 'bo-')
# plt.vlines(3, ymin=min(inertia)*0.9999, ymax=max(inertia)*1.0003, linestyles='--', colors='g', label='k=3')
# plt.vlines(4, ymin=min(inertia)*0.9999, ymax=max(inertia)*1.0003, linestyles='--', colors='r', label='k=4')

# plt.xlabel("k")
# plt.ylabel("Inertia")
# plt.title("Elbow Method")
# plt.legend()
# plt.show()

# PCA (2차원 축소)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 1. K=3 클러스터링
kmeans3 = KMeans(n_clusters=3, random_state=42)
labels3 = kmeans3.fit_predict(X_scaled)

# 2. K=4 클러스터링
kmeans4 = KMeans(n_clusters=4, random_state=42)
labels4 = kmeans4.fit_predict(X_scaled)

# 시각화
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
plt.show() 

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}, Silhouette Score={score:.4f}")
# k=3 선택