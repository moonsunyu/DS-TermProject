# 최종 모델
# 파생 변수('roi') 추가 + target encoding

# decisionTree3에 target encoding까지 추가
# 감독, 배우, 작가가 영화 흥행에 미치는 영향을 알 수 있음

import os
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

# 저장 경로 설정
save_dir = "results/learning-models/decisionTree4"
os.makedirs(save_dir, exist_ok=True)

# 1. 데이터 로드 및 전처리
df = pd.read_csv("data/feature-engineered/movies_clustered.csv")
df = df.drop(columns=["name", "cluster_label", "votes"])

# 2. ROI 파생 변수 추가
df["roi"] = df["gross"] / df["budget"]
df["roi"].replace([np.inf, -np.inf], np.nan, inplace=True)
df["roi"].fillna(df["roi"].median(), inplace=True)
df = df.drop(columns=["gross", "budget"])

# 3. Target Encoding: star_top30, director_top10, writer_top10
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
categorical_cols_to_encode = ['rating', 'runtime_category', 'genre_top10',
                               'country_top5', 'company_top10', 'cluster']
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

# 8. 모델 학습 (Decision Tree)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# 9. 교차검증 평가
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores), "Std:", np.std(cv_scores))

# 10. 테스트셋 평가
y_pred = clf.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Report.txt 저장
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

with open(os.path.join(save_dir, "Report.txt"), "w") as f:
    f.write("Cross-validation scores: " + str(cv_scores) + "\n")
    f.write(f"Mean accuracy: {np.mean(cv_scores)} Std: {np.std(cv_scores)}\n\n")
    f.write("Confusion Matrix:\n" + str(conf_matrix) + "\n\n")
    f.write("Classification Report:\n" + class_report)

# 11. 특성 중요도 시각화
feature_importances = clf.feature_importances_
feature_names = X_train.columns

# 그룹 매핑
categorical_cols = [
    'rating', 'runtime_category', 'genre_top10',
    'country_top5', 'company_top10', 'cluster'
]
category_map = {}
for name in feature_names:
    matched = False
    for cat in categorical_cols:
        if name.startswith(cat + '_'):
            category_map[name] = cat
            matched = True
            break
    if not matched:
        category_map[name] = name 

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
})
importance_df['group'] = importance_df['feature'].map(category_map)
grouped_importance = importance_df.groupby('group')['importance'].sum().sort_values(ascending=False).head(20)

# 시각화
plt.figure(figsize=(12, 6))
colors = sns.color_palette("magma", len(grouped_importance))
sns.barplot(
    x=grouped_importance.values,
    y=grouped_importance.index,
    palette=colors
)
plt.xscale("log")
plt.gca().xaxis.set_major_locator(LogLocator(base=10.0))
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$"))
plt.title("Total Feature Importance by Category (Log-Scaled Axis)", fontsize=14)
plt.xlabel("Importance (Log Scale)", fontsize=12)
plt.ylabel("Feature Group", fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Feature_Importance.png"))
plt.show()


# 12. 트리 시각화
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=["Not Hit", "Hit"],
    filled=True,
    impurity=True,
    rounded=True,
    max_depth=4,
    fontsize=8
)
plt.title("Decision Tree Visualization", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Plot_Tree.png"))
plt.show() 