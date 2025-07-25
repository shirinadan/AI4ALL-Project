import pandas as pd
import numpy as np

# Load the cleaned dataset
input_path = "data/cleaned_us_data.csv"
print(f"Loading cleaned dataset from {input_path}...")
df = pd.read_csv(input_path)
print(f"Original shape: {df.shape}")

# 1. Drop ID/meta/date columns
drop_cols = [
    "Unnamed: 0",
    "id",
    "object_id",
    "name",
    "city",
    "zip_code",
    "founded_at",
    "closed_at",
    "first_funding_at",
    "last_funding_at",
    "state_code",
    "state_code.1",
    "Unnamed: 6",
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# 2. Separate target variable
target_col = "success"
y = df[target_col]
X = df.drop(columns=[target_col])

# 3. Drop near-constant columns
from sklearn.feature_selection import VarianceThreshold

# Drop features with 0 or near-0 variance
selector = VarianceThreshold(threshold=0.01)
X_reduced = selector.fit_transform(X)

# Get kept feature names
selected_columns = X.columns[selector.get_support()]

# Combine back with target
df_selected = pd.DataFrame(X_reduced, columns=selected_columns)
df_selected[target_col] = y.values

# 4. Save reduced feature set
output_path = "data/selected_us_data.csv"
df_selected.to_csv(output_path, index=False)
print(f"✅ Selected features saved to {output_path}")
print(f"New shape: {df_selected.shape}")
