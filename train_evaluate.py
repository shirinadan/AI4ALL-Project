import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

# === Load Data ===
df = pd.read_csv("data/selected_us_data.csv")
print(f"✅ Loaded dataset. Shape: {df.shape}")

# === Define target and drop leakage ===
y = df["success"]
X = df.drop(columns=["success", "labels"], errors="ignore")  # remove label leakage

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# === Balance classes with SMOTE ===
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(
    f"Balanced Train shape: {X_train_bal.shape}, Class distribution: {np.bincount(y_train_bal)}"
)

# === Train model with GridSearch ===
param_grid = {
    "learning_rate": [0.05, 0.1],
    "max_iter": [100, 200],
    "max_depth": [3, 5, None],
}
model = HistGradientBoostingClassifier(random_state=42)
grid = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train_bal, y_train_bal)
best_model = grid.best_estimator_

# === Evaluate on test set ===
y_pred = best_model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("=== Accuracy ===")
print(f"{accuracy_score(y_test, y_pred):.4f}")

print("=== Precision ===")
print(f"{precision_score(y_test, y_pred):.4f}")

print("=== Recall ===")
print(f"{recall_score(y_test, y_pred):.4f}")

print("=== F1 Score ===")
print(f"{f1_score(y_test, y_pred):.4f}")

print("=== ROC-AUC ===")
print(f"{roc_auc_score(y_test, y_pred):.4f}")

# === Feature Importance ===
print("\n🔍 Calculating Permutation Feature Importance...")
result = permutation_importance(
    best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
sorted_idx = result.importances_mean.argsort()[::-1]
top_n = 15
top_features = X_test.columns[sorted_idx[:top_n]]
top_importances = result.importances_mean[sorted_idx[:top_n]]

# === Plot Feature Importance ===
plt.figure(figsize=(10, 6))
plt.barh(top_features[::-1], top_importances[::-1])
plt.xlabel("Permutation Importance")
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()
