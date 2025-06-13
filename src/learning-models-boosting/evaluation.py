import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# 1. Load and preprocess data
df = pd.read_csv("data/feature-engineered/movies_clustered.csv")
df = df.drop(columns=["name", "cluster_label", "votes"])

df["roi"] = df["gross"] / df["budget"]
df["roi"].replace([np.inf, -np.inf], np.nan, inplace=True)
df["roi"].fillna(df["roi"].median(), inplace=True)
df = df.drop(columns=["gross", "budget"])

X = df.drop(columns=["is_hit"])
y = df["is_hit"]

X_encoded = pd.get_dummies(X, drop_first=True)

numeric_cols = ['score', 'runtime', 'log_votes', 'weighted_score', 'roi']
scaler = StandardScaler()
X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 2. Define models
models = {
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=10, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# 3. Evaluation loop with console output
for name, model in models.items():
    print(f"\n\n==== {name} ====")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_encoded, y, cv=cv, scoring='accuracy')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}, Std: {np.std(cv_scores):.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

# 4. ROC Curve visualization
plt.figure(figsize=(10, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()