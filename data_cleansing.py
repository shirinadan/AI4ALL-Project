# -*- coding: utf-8 -*-
"""
Data Cleansing and Exploration of Global Startup Success Dataset

This script loads the Global Startup Success Dataset, performs cleaning operations,
feature engineering, outlier handling, scaling, and feature selection.
"""

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import os
import datetime
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from kagglehub import KaggleDatasetAdapter

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
)
from sklearn.feature_selection import mutual_info_regression, SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# =============================================================================
# 2. INITIAL SETUP & DATA LOADING
# =============================================================================
# Set pandas display option to show all columns
pd.set_option("display.max_columns", None)

# Load the dataset using the older KaggleHub function to avoid environment issues
# A DeprecationWarning is expected and can be ignored for now.
print("Loading dataset...")
file_path = "global_startup_success_dataset.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "hamnakaleemds/global-startup-success-dataset",
    file_path,
)
print("Dataset loaded successfully.")
print("-" * 50)


# =============================================================================
# 3. INITIAL DATA INSPECTION
# =============================================================================
print("First 5 records of the dataset:")
print(df.head())
print("\n" + "=" * 50 + "\n")

# Check for and remove duplicates
print(f"Number of rows before removing duplicates: {len(df)}")
df.drop_duplicates(inplace=True)
print(f"Number of rows after removing duplicates: {len(df)}")
print("\n" + "=" * 50 + "\n")


# =============================================================================
# 4. DATA CLEANING AND PREPROCESSING
# =============================================================================
print("--- Cleaning and Preprocessing ---")

# Convert 'Startup Name' to a numeric 'StartupID'
df["StartupID"] = (
    df["Startup Name"].str.extract(r"_(\d+)", expand=False).astype("Int64")
)
df = df.drop(columns=["Startup Name"], errors="ignore")
# Move StartupID to the first column
if "StartupID" in df.columns:
    cols = list(df.columns)
    cols.remove("StartupID")
    cols.insert(0, "StartupID")
    df = df[cols]
print("'StartupID' created and moved to the front.")
print("\n" + "=" * 30 + "\n")

# Convert binary categorical columns ('Acquired?', 'IPO?') to numeric (0/1)
binary_map = {"Yes": 1, "No": 0}
df["Acquired?"] = df["Acquired?"].map(binary_map)
df["IPO?"] = df["IPO?"].map(binary_map)
print("Binary columns converted.")
print("\n" + "=" * 30 + "\n")


# =============================================================================
# 4.5 FEATURE ENGINEERING
# =============================================================================
print("--- Feature Engineering ---")

# 1. Create 'Startup Age'
current_year = datetime.datetime.now().year
df["Startup Age"] = current_year - df["Founded Year"]
print("'Startup Age' feature created.")

# 2. Engineer Financial Ratios
df["Number of Employees"] = df["Number of Employees"].replace(0, np.nan)
df["Funding per Employee"] = df["Total Funding ($M)"] / df["Number of Employees"]
df["Revenue per Employee"] = df["Annual Revenue ($M)"] / df["Number of Employees"]
df[["Funding per Employee", "Revenue per Employee"]] = df[
    ["Funding per Employee", "Revenue per Employee"]
].fillna(0)
df.replace([np.inf, -np.inf], 0, inplace=True)
print("Financial ratio features created.")

# 3. Advanced Encoding for Categorical Features
country_freq = df["Country"].value_counts(normalize=True)
df["Country_Encoded"] = df["Country"].map(country_freq)

cols_to_encode = ["Funding Stage", "Industry"]
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_data = ohe.fit_transform(df[cols_to_encode])
encoded_df = pd.DataFrame(
    encoded_data, columns=ohe.get_feature_names_out(cols_to_encode)
)
df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
print("Categorical features encoded.")

# 4. Extract Information from 'Tech Stack'
df["Uses_Python"] = (
    df["Tech Stack"].str.contains("Python", case=False, na=False).astype(int)
)
df["Uses_Java"] = (
    df["Tech Stack"].str.contains("Java", case=False, na=False).astype(int)
)
df["Uses_Nodejs"] = (
    df["Tech Stack"].str.contains("Node.js", case=False, na=False).astype(int)
)
df["Uses_AI"] = df["Tech Stack"].str.contains("AI", case=False, na=False).astype(int)
print("Tech Stack features extracted.")

# 5. Drop original columns that have been transformed
df = df.drop(
    columns=["Country", "Funding Stage", "Industry", "Tech Stack"], errors="ignore"
)
print("Original transformed columns dropped.")
print("\n" + "=" * 50 + "\n")


# =============================================================================
# 4.6 LOG TRANSFORMATION FOR SKEWED DATA
# =============================================================================
print("--- Log Transformation for Skewed Data ---")
skewed_cols = [
    "Total Funding ($M)",
    "Number of Employees",
    "Annual Revenue ($M)",
    "Valuation ($B)",
    "Customer Base (Millions)",
    "Social Media Followers",
    "Funding per Employee",
    "Revenue per Employee",
]
log_skewed_cols = []

for col in skewed_cols:
    # Check if column exists before trying to transform it
    if col in df.columns:
        df[col] = np.log1p(df[col])
        new_col_name = f"Log_{col}"
        df.rename(columns={col: new_col_name}, inplace=True)
        log_skewed_cols.append(new_col_name)

print("Skewed data has been log-transformed.")
print("\n" + "=" * 50 + "\n")


# =============================================================================
# 4.7 OUTLIER DETECTION AND HANDLING (IQR METHOD)
# =============================================================================
print("--- Outlier Detection and Handling using IQR ---")
# Only apply outlier detection to the continuous log-transformed columns
outlier_mask = pd.Series(False, index=df.index)

for col in log_skewed_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = outlier_mask | (df[col] < lower_bound) | (df[col] > upper_bound)

print(f"Number of outlier rows found: {outlier_mask.sum()}")
df_cleaned = df[~outlier_mask].copy()
print(f"Original DataFrame shape: {df.shape}")
print(f"Cleaned DataFrame shape after removing outliers: {df_cleaned.shape}")
print("\n" + "=" * 50 + "\n")


# =============================================================================
# 5. FEATURE SCALING (STANDARDIZATION)
# =============================================================================
# Add a check to prevent scaling an empty dataframe
if not df_cleaned.empty:
    print("--- Feature Scaling ---")
    scaler = StandardScaler()
    features_to_scale = df_cleaned.select_dtypes(include=np.number).columns.drop(
        ["StartupID", "Success Score"]
    )
    df_cleaned[features_to_scale] = scaler.fit_transform(df_cleaned[features_to_scale])

    print("Numeric features have been standardized.")
else:
    print(
        "Skipping Feature Scaling because the dataframe is empty after outlier removal."
    )
print("\n" + "=" * 50 + "\n")


# =============================================================================
# 6. FEATURE SELECTION
# =============================================================================
if not df_cleaned.empty:
    print("--- Feature Selection ---")
    # Define features (X) and target (y)
    X = df_cleaned.drop(columns=["StartupID", "Success Score"])
    y = df_cleaned["Success Score"]

    # --- VISUALIZATION TO HELP DECIDE ON FEATURES ---

    # 1. Mutual Information Scores
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    # 2. Correlation with Target
    correlations = X.corrwith(y).abs().sort_values(ascending=False)

    # Create plots
    plt.figure(figsize=(20, 8))

    # Plot for Mutual Information
    plt.subplot(1, 2, 1)
    mi_scores.head(15).plot(kind="barh", color="skyblue")
    plt.title("Top 15 Features by Mutual Information")
    plt.xlabel("Mutual Information Score")

    # Plot for Correlation
    plt.subplot(1, 2, 2)
    correlations.head(15).plot(kind="barh", color="salmon")
    plt.title("Top 15 Features by Correlation with Success Score")
    plt.xlabel("Absolute Correlation")

    plt.tight_layout()
    plt.show()

    print("\nTop 10 features based on Mutual Information:")
    print(mi_scores.head(10))
    print("\nTop 10 features based on Correlation:")
    print(correlations.head(10))

else:
    print("Skipping Feature Selection because the dataframe is empty.")
print("\n" + "=" * 50 + "\n")


# =============================================================================
# 7. FINAL DATA PREPARATION COMPLETE
# =============================================================================
print("Data preparation, feature engineering, and selection steps are complete.")
if not df_cleaned.empty:
    print("The 'df_cleaned' DataFrame is ready for machine learning.")
else:
    print(
        "Warning: The 'df_cleaned' DataFrame is empty. Review outlier detection parameters."
    )
