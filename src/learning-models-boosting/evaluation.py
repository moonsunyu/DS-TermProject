# Evaluation
# RandomForest, XGBoost

import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.ticker import LogLocator, FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# 결과 저장 디렉토리
base_dir = "results/evaluation"
os.makedirs(base_dir, exist_ok=True)

# 1. 데이터 로드 및 전처리
df = pd.read_csv("data/feature-engineered/movies_clustered.csv")
df = df.drop(columns=["name", "cluster_label", "votes"])

# 2. ROI 파생 변수 추가
df["roi"] = df["gross"] / df["budget"]
df["roi"].replace([np.inf, -np.inf], np.nan, inplace=True)
df["roi"].fillna(df["roi"].median(), inplace=True)
df = df.drop(columns=["gross", "budget"])

# 3. Target Encoding: star_top30, director_top10
star_mean_map = df.groupby("star_top30")["is_hit"].mean()
director_mean_map = df.groupby("director_top10")["is_hit"].mean()
writer_mean_map = df.groupby("writer_top10")["is_hit"].mean()

df["star_encoded"] = df["star_top30"].map(star_mean_map)         
df["director_encoded"] = df["director_top10"].map(director_mean_map)  
df["writer_encoded"] = df["writer_top10"].map(writer_mean_map)  
df = df.drop(columns=["star_top30", "director_top10", "writer_top10"])

# 4. 타겟 분리 및 학습/테스트 분할
y = df["is_hit"]
X = df.drop(columns=["is_hit"])
X["star_roi_interaction"] = X["star_encoded"] * X["roi"]
X["director_score_interaction"] = X["director_encoded"] * X["score"]
X["writer_score_interaction"] = X["writer_encoded"] * X["score"]


# 5. 범주형 인코딩 (One-hot encoding 적용 대상만)
categorical_cols_to_encode = ['rating', 'runtime_category', 
                              'genre_top10', 'country_top5', 'company_top10', 'cluster']
X = pd.get_dummies(X, columns=categorical_cols_to_encode, drop_first=True)

# 6. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 스케일링
numeric_cols = [
    'score', 'runtime', 'log_votes', 'weighted_score',
    'roi', 'star_encoded', 'director_encoded', 'writer_encoded',
    'star_roi_interaction', 'director_score_interaction', 'writer_score_interaction'
]
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=10, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

for name, model in models.items():
    print(f"\n =={name}== \n")
    
    # 학습
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 정확도
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # 교차검증
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")
    
    # 혼동 행렬 및 리포트
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# 저장할 예측 확률
prob_preds = {}

# 모델별 평가 및 저장
for name, model in models.items():
    print(f"\n== {name} ==")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # 저장 폴더 생성
    model_dir = os.path.join(base_dir, name)
    os.makedirs(model_dir, exist_ok=True)

    # 평가 결과 저장
    with open(os.path.join(model_dir, "report.txt"), "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Cross-validation scores: {cv_scores}\n")
        f.write(f"Mean CV accuracy: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 예측 확률 계산
y_pred_dt_proba = models["Decision Tree"].predict_proba(X_test)[:, 1]
y_pred_rf_proba = models["Random Forest"].predict_proba(X_test)[:, 1]
y_pred_xgb_proba = models["XGBoost"].predict_proba(X_test)[:, 1]

# FPR, TPR 계산
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf_proba)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb_proba)

# AUC 계산
roc_auc_dt = auc(fpr_dt, tpr_dt)
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# 그래프 시각화
plt.figure(figsize=(10, 6))
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.3f})', lw=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})', lw=2)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.3f})', lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "roc_curve_comparison.png"))
plt.show() 