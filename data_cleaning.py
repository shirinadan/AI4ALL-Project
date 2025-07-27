import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# === Load Dataset ===
file_path = "data/startup_crunchbase.csv"  # replace with your actual file name
print("✅ Dataset loaded.")
df = pd.read_csv(file_path, encoding="ISO-8859-1")  # fix for unicode errors

# === Standardize Column Names ===
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Shape:", df.shape)

# === Display Sample ===
print("\n=== First 5 rows ===")
print(df.head())

# === Column Info ===
print("\n=== Column Info ===")
print(df.info())

# === Missing Values ===
print("\n=== Missing Values ===")
print(df.isnull().sum())

# === Descriptive Statistics ===
print("\n=== Descriptive Stats ===")
print(df.describe(include="all"))

# === Clean Funding Column ===
if "funding_total_usd" in df.columns:
 # Step 1: Remove any characters that are not digits or dots
    df['funding_total_usd'] = (
        df['funding_total_usd']
        .astype(str)
        .str.strip()
        .str.replace(r'[^\d.]', '', regex=True)
    )

# Step 2: Convert to float, forcing errors to NaN
df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')



# === Clean Funding Rounds ===
if "funding_rounds" in df.columns:
    df["funding_rounds"] = pd.to_numeric(df["funding_rounds"], errors="coerce")

# === Clean Founded Year ===
if "founded_year" in df.columns:
    df["founded_year"] = pd.to_numeric(df["founded_year"], errors="coerce")

# === Drop rows with missing critical fields ===
df = df.dropna(subset=["funding_total_usd", "funding_rounds"])

# === Success Score (scaled) ===
# Scale both funding and rounds to [0, 1], then weight them
df["funding_scaled"] = (df["funding_total_usd"] - df["funding_total_usd"].min()) / (
    df["funding_total_usd"].max() - df["funding_total_usd"].min()
)
df["rounds_scaled"] = (df["funding_rounds"] - df["funding_rounds"].min()) / (
    df["funding_rounds"].max() - df["funding_rounds"].min()
)

# Compute final success score on scale of 1 to 5
df["success_score"] = 1 + 4 * (0.7 * df["funding_scaled"] + 0.3 * df["rounds_scaled"])

# === Display Score Summary ===
print("\n=== Success Score Summary ===")
print(df["success_score"].describe())

# === Histograms ===
cols_to_plot = ["funding_total_usd", "funding_rounds", "success_score"]
for col in cols_to_plot:
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# === Save cleaned data ===
df.to_csv("cleaned_crunchbase_with_success_score.csv", index=False)
print("✅ Cleaned dataset saved as 'cleaned_crunchbase_with_success_score.csv'")
