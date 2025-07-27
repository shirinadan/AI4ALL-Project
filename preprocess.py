import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned Crunchbase dataset
df = pd.read_csv("cleaned_crunchbase_with_success_score.csv")

# Step 1: Drop rows with missing success score
df = df.dropna(subset=["success_score"])

# Step 2: Log-transform skewed columns
df["funding_total_usd_log"] = np.log1p(df["funding_total_usd"])

# Optional: Replace original score with log-transformed version if it helps modeling
# df['success_score'] = np.log1p(df['success_score'])

# Step 3: Select categorical columns to one-hot encode
categorical_cols = ["market", "status", "country_code"]
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

df = pd.get_dummies(df, columns=categorical_cols)

# Step 4: Drop unnecessary or string-heavy columns
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
    "funding_total_usd",
]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Step 5: Save cleaned data
df.to_csv("prepared_crunchbase_for_regression.csv", index=False)
print("✅ Preprocessed dataset saved as 'prepared_crunchbase_for_regression.csv'")

# Step 6: Show correlation heatmap with success_score
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
