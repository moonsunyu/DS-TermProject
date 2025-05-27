import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# 1. 데이터 로드
df = pd.read_csv("data/feature-engineered/movies_clustered.csv")
df_model = df.drop(columns=['name', 'cluster', 'gross', 'budget', 'votes'])

# 2. 피처 설정
numeric_cols = ['score', 'log_votes', 'runtime', 'weighted_score']
categorical_cols = [
    'rating', 'runtime_category', 'director_top10', 'writer_top10',
    'star_top30', 'genre_top10', 'country_top5', 'company_top10', 'cluster_label'
]

# 3. 인코딩
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# 4. X, y 정의
X = df_encoded.drop(columns=['is_hit'])
y = df_encoded['is_hit']

# 5. 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 모델 학습
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# 7. 예측 및 평가
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# 10. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_score)
ap_score = average_precision_score(y_test, y_score)

# 11. Feature Importance 그룹 시각화
importances = clf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

category_map = {}
for feature in feature_importance_df['feature']:
    matched = False
    for cat in categorical_cols:
        if feature.startswith(cat + '_'):
            category_map[feature] = cat
            matched = True
            break
    if not matched:
        category_map[feature] = feature  # numeric feature

feature_importance_df['group'] = feature_importance_df['feature'].map(category_map)
grouped_importance = feature_importance_df.groupby('group')['importance'].sum().sort_values(ascending=False)


# Feature Importance (log scale)
sns.barplot(x=grouped_importance.values, y=grouped_importance.index, palette='magma')
plt.xscale('log')
plt.title("Total Feature Importance by Category (Log Scale)")
plt.xlabel("Log(Importance)")
plt.ylabel("Feature Group")

plt.show()


# 가장 유력하다!