# decisionTree_3.py
# Instead of simply removing strong features, a derived feature 'roi' (return on investment) was introduced as a replacement

# The model performs well, but the feature importance results include several values that are not intuitively reasonable

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# Save path
save_dir = "results/learning-models/decisionTree3"
os.makedirs(save_dir, exist_ok=True)

# 1. Data load and preprocessing
df = pd.read_csv("data/feature-engineered/movies_clustered.csv")
df = df.drop(columns=["name", "cluster_label", "votes"]) 

# 2. Add ‘roi’ (=‘gross’/’budget’)
df["roi"] = df["gross"] / df["budget"]
df["roi"].replace([np.inf, -np.inf], np.nan, inplace=True)
df["roi"].fillna(df["roi"].median(), inplace=True)
df = df.drop(columns=["gross", "budget"]) # Remove and replace strong columns

# 3. Separate targets and features
X = df.drop(columns=["is_hit"])
y = df["is_hit"]

# 4. Categorical Encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# 5. Numerical Feature scaling
numeric_cols = ['score', 'runtime', 'log_votes', 'weighted_score', 'roi'] # votes is excluded because it is duplicated with log_votes
scaler = StandardScaler()
X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

# 6. Training/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 7.DecisionTree Learning
clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# 8. Cross-validation evaluation and test set evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_encoded, y, cv=cv, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores), "Std:", np.std(cv_scores))

y_pred = clf.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Report.txt
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

with open(os.path.join(save_dir, "Report.txt"), "w") as f:
    f.write("Cross-validation scores: " + str(cv_scores) + "\n")
    f.write(f"Mean accuracy: {np.mean(cv_scores)} Std: {np.std(cv_scores)}\n\n")
    f.write("Confusion Matrix:\n" + str(conf_matrix) + "\n\n")
    f.write("Classification Report:\n" + class_report)

# Feature importance
feature_importances = clf.feature_importances_
feature_names = X_encoded.columns

categorical_cols = [
    'rating', 'runtime_category', 'director_top10', 'writer_top10',
    'star_top30', 'genre_top10', 'country_top5', 'company_top10', 'cluster'
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


plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=X_encoded.columns,
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
