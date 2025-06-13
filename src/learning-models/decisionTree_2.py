# decisionTree_2.py
# Train a basic model after dropping 'gross' and 'budget' from the preprocessed dataset

# The issue of feature dominance was resolved,
# but model accuracy dropped significantly, making it impractical to use.


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Save path
save_dir = "results/learning-models/decisionTree2"
os.makedirs(save_dir, exist_ok=True)

# 1. 데이터 로드
df = pd.read_csv("data/feature-engineered/movies_clustered.csv")
df_model = df.drop(columns=['name', 'cluster_label', 'votes', 'gross', 'budget']) # gross, budget 모두 drop

# 2. 피처 설정
numeric_cols = ['score', 'log_votes', 'runtime', 'weighted_score']
categorical_cols = [
    'rating', 'runtime_category', 'director_top10', 'writer_top10',
    'star_top30', 'genre_top10', 'country_top5', 'company_top10'
]

# 3. 인코딩
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# 4. X, y 정의
X = df_encoded.drop(columns=['is_hit'])
y = df_encoded['is_hit']

# 5. 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 학습
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# 7. 교차검증 평가 테스트셋 평가
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores), "Std:", np.std(cv_scores))

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

# 8. 중요도 추출
importances = clf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# 9. 그룹 매핑
category_map = {}
for feature in feature_importance_df['feature']:
    matched = False
    for cat in categorical_cols:
        if feature.startswith(cat + '_'):
            category_map[feature] = cat
            matched = True
            break
    if not matched:
        category_map[feature] = feature  

feature_importance_df['group'] = feature_importance_df['feature'].map(category_map)

# 10. 그룹별 중요도 합산
grouped_importance = feature_importance_df.groupby('group')['importance'].sum().sort_values(ascending=False)

# 11. 시각화 (log scale)
plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_importance.values, y=grouped_importance.index, palette='magma')
plt.xscale('log')
plt.title("Total Feature Importance by Category (Log Scale)")
plt.xlabel("Log(Importance)")
plt.ylabel("Feature Group")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Feature_Importance.png"))
plt.show()

# 트리 시각화 (상위 분기만 표시)
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=X.columns,
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
