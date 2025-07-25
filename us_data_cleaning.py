import pandas as pd
import numpy as np

# Set display options
pd.set_option("display.max_columns", None)

# Load the CSV file (no need for kagglehub anymore since you downloaded it with the CLI)
print("Loading dataset...")
file_path = "data/startup_data.csv"  # Correct relative path and filename
df = pd.read_csv(file_path)
print(f"Initial shape: {df.shape}")

# === Drop columns that won't help the model ===
df.drop(
    columns=[
        "Unnamed: 0",
        "Unnamed: 6",
        "id",
        "object_id",
        "state_code.1",
        "name",
        "city",
        "zip_code",
    ],
    inplace=True,
)

# === Drop rows with critical missing values ===
df.dropna(subset=["age_first_milestone_year", "age_last_milestone_year"], inplace=True)

# === Fill missing closed_at ===
df["closed_at"] = df["closed_at"].fillna("N/A")

# === Binary encode target variable (status) ===
# We treat "acquired" as success = 1, others as 0
df["success"] = (df["status"].str.lower() == "acquired").astype(int)
df.drop(columns=["status"], inplace=True)

# === One-hot encode 'category_code' ===
df = pd.get_dummies(df, columns=["category_code"], prefix="cat", drop_first=True)

# === Drop date string columns if not used ===
df.drop(
    columns=["founded_at", "first_funding_at", "last_funding_at", "closed_at"],
    inplace=True,
)

# === Final check ===
print("\n✅ Cleaned shape:", df.shape)
print(df.head())
output_path = "data/cleaned_data.csv"
df.to_csv(output_path, index=False)
