import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# feature engineering된 데이터셋 (clustering 전)
df = pd.read_csv("data/feature-engineered/movies_normalized.csv") 

# Clustering: k-means
# 영화 데이터를 유형별로 분류해서 숨겨진 패턴을 찾아내기 위한 기법
cluster_features = ['budget', 'weighted_score']
X = df[cluster_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (2차원 축소)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 1. K=3 클러스터링
kmeans3 = KMeans(n_clusters=3, random_state=42)
labels3 = kmeans3.fit_predict(X_scaled)

# # 시각화
# plt.figure(figsize=(12, 5))
# # K=3
# plt.subplot(1, 2, 1)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels3, cmap='viridis')
# plt.title("K-Means Clustering (k=3)")
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.show() 

# KMeans 클러스터링 (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 클러스터별 평균값 계산
cluster_summary = df.groupby('cluster')[['budget', 'score', 'gross', 'is_hit']].mean().round(2)
cluster_summary['count'] = df['cluster'].value_counts().sort_index()


# 클러스터 이름 붙이기
# 클러스터별 특성을 보고 수동으로 이름을 붙여줌

# 분위수 확인
print(df['budget'].describe())

cluster_names = {}
for cluster in cluster_summary.index:
    row = cluster_summary.loc[cluster]
    budget = row['budget']
    hit_rate = row['is_hit']

    if budget >= 0.78:    # 상위 20%
        if hit_rate >= 0.5:
                name = "high-budget-success"
        else:
                name = "high-budget-failure"   
    elif budget < 0.70:          # 하위 20%
        if hit_rate >= 0.5:
                name = "low-budget-success"
        else:
                name = "low-budget-failure" 
    else:
        name = "mid-budget"

    cluster_names[cluster] = name 

# 이름 매핑 추가
df['cluster_label'] = df['cluster'].map(cluster_names)

# 결과 출력
print("Average Summary by Cluster:")
print(cluster_summary)
print("\nMapping Cluster Names:")
for k, v in cluster_names.items():
    print(f"Cluster {k}: {v}")


df.to_csv("data/feature-engineered/movies_clustered.csv", index=False)

print("데이터셋 저장 완료: data/feature-engineered/movies_clustered.csv")