import pandas as pd
from sklearn.ensemble import IsolationForest

# 데이터 불러오기
df = pd.read_csv("data/cleaned/movies_cleaned.csv")

# 수치형 feature 선택
features = ['budget', 'gross', 'score', 'votes', 'runtime']
df_numeric = df[features].copy()

# Isolation Forest로 이상치 탐지
iso = IsolationForest(contamination=0.01, random_state=42)
outlier_pred = iso.fit_predict(df_numeric)

# 이상치 제거
df_cleaned_outliers = df[outlier_pred != -1]  # 정상치만 남김

# (결과 저장
df_cleaned_outliers.to_csv("data/cleaned/movies_cleaned_outliers.csv", index=False)

print("데이터셋 저장 완료: data/cleaned/movies_cleaned_outliers.csv")
