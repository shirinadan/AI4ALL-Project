import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

# === Load selected dataset ===
df = pd.read_csv("data/selected_us_data.csv")

# === Define target and features ===
y = df["success"]

X = df.drop(columns=["success"])
# Drop potential target leakage column
if "labels" in X.columns:
    X = X.drop(columns=["labels"])

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# === Train model ===
model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# === Compute feature importances using permutation ===
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# === Extract and sort importances ===
importances = result.importances_mean
indices = np.argsort(importances)[::-1][:20]  # Top 20

top_features = X.columns[indices]
top_importances = importances[indices]

# === Plot ===
plt.figure(figsize=(10, 6))
plt.barh(top_features[::-1], top_importances[::-1])
plt.xlabel("Permutation Importance")
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()
