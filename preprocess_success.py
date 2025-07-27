import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# === Load Dataset ===
file_path = "data/startup_crunchbase.csv"  # change to your actual path
print("✅ Loading dataset...")
df = pd.read_csv(file_path, encoding="ISO-8859-1")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("✅ Shape:", df.shape)

# === Clean funding_total_usd ===
if "funding_total_usd" in df.columns:
    df["funding_total_usd"] = (
        df["funding_total_usd"]
        .astype(str)
        .str.strip()
        .str.replace(r"[^\d.]", "", regex=True)
    )
    df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"], errors="coerce")

# === Clean funding_rounds ===
if "funding_rounds" in df.columns:
    df["funding_rounds"] = pd.to_numeric(df["funding_rounds"], errors="coerce")

# === Drop rows with missing values required for scoring ===
df = df.dropna(subset=["funding_total_usd", "funding_rounds"])

# === Redefine Success Score with fewer features ===
print("📈 Calculating success score (funding + rounds)...")
df["funding_scaled"] = (df["funding_total_usd"] - df["funding_total_usd"].min()) / (
    df["funding_total_usd"].max() - df["funding_total_usd"].min()
)
df["rounds_scaled"] = (df["funding_rounds"] - df["funding_rounds"].min()) / (
    df["funding_rounds"].max() - df["funding_rounds"].min()
)
df["success_score"] = 1 + 4 * (0.7 * df["funding_scaled"] + 0.3 * df["rounds_scaled"])

# === Drop score-related features to avoid leakage ===
features_to_drop = [
    "funding_total_usd",
    "funding_rounds",
    "funding_scaled",
    "rounds_scaled",
]
df.drop(columns=features_to_drop, inplace=True, errors="ignore")

# === Encode categorical columns ===
categorical_cols = ["market", "status", "country_code"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")
df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns])

# === Drop text-heavy, non-predictive columns ===
drop_cols = [
    "permalink",
    "name",
    "homepage_url",
    "category_list",
    "region",
    "city",
    "founded_at",
    "founded_month",
    "founded_quarter",
    "first_funding_at",
    "last_funding_at",
]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# === Save Final Cleaned Dataset ===
out_path = "prepared_crunchbase_for_regression.csv"
df.to_csv(out_path, index=False)
print(f"✅ Final cleaned dataset saved as '{out_path}'")

# === Correlation with success_score ===
plt.figure(figsize=(12, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(
    corr[["success_score"]].sort_values("success_score", ascending=False),
    annot=True,
    cmap="coolwarm",
)
plt.title("Correlation with Success Score")
plt.tight_layout()
plt.show()
