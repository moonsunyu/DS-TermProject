import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# 1. 데이터 불러오기
df = pd.read_csv("data/normalized/movies_preprocessed_normalized.csv")

# 2. 문자열 변수 인코딩
categorical_cols = [
    "rating", "runtime_category", "director_top10", "writer_top10",
    "star_top30", "genre_top10", "country_top5", "company_top10"
]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 3. feature / label 분리
drop_cols = ["is_hit", "gross", "name", "cluster_label"]
X = df.drop(columns=drop_cols)
y = df["is_hit"]

# 4. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. 모델 학습
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 6. 교차검증
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
print(f"5-Fold CV 평균 정확도: {cv_scores.mean():.4f}")

# 7. 테스트셋 평가
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, digits=4)
print(report)

# 8. 결과 저장
os.makedirs("results/model-report", exist_ok=True)
with open("results/model-report/decision_report.txt", "w") as f:
    f.write(report)

# 9. 모델 저장
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/decision_tree_model.pkl")
print("모델 저장 완료: model/decision_tree_model.pkl")
