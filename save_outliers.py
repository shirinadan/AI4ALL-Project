# -*- coding: utf-8 -*-
"""
Data Cleansing and Exploration of Global Startup Success Dataset

This script prepares two final datasets for model training:
1.  A feature-engineered dataset WITH outliers.
2.  A feature-engineered dataset WITHOUT outliers.
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
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split


# =============================================================================
# 2. INITIAL SETUP & DATA LOADING
# =============================================================================
pd.set_option("display.max_columns", None)
print("Loading dataset...")
file_path = "global_startup_success_dataset.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "hamnakaleemds/global-startup-success-dataset",
    file_path,
)
print("Dataset loaded successfully.")
df.drop_duplicates(inplace=True)


# =============================================================================
# 3. DATA CLEANING AND FEATURE ENGINEERING (Condensed)
# =============================================================================
print("--- Starting Data Cleaning and Feature Engineering ---")
# Convert 'Startup Name' to a numeric 'StartupID'
df["StartupID"] = (
    df["Startup Name"].str.extract(r"_(\d+)", expand=False).astype("Int64")
)
df = df.drop(columns=["Startup Name"], errors="ignore")

# Convert binary categorical columns
binary_map = {"Yes": 1, "No": 0}
df["Acquired?"] = df["Acquired?"].map(binary_map)
df["IPO?"] = df["IPO?"].map(binary_map)

# Engineer new features
current_year = datetime.datetime.now().year
df["Startup Age"] = current_year - df["Founded Year"]
df["Number of Employees"] = df["Number of Employees"].replace(0, np.nan)
df["Funding per Employee"] = df["Total Funding ($M)"] / df["Number of Employees"]
df["Revenue per Employee"] = df["Annual Revenue ($M)"] / df["Number of Employees"]
df[["Funding per Employee", "Revenue per Employee"]] = df[
    ["Funding per Employee", "Revenue per Employee"]
].fillna(0)
df.replace([np.inf, -np.inf], 0, inplace=True)

# Encode categorical features
country_freq = df["Country"].value_counts(normalize=True)
df["Country_Encoded"] = df["Country"].map(country_freq)
cols_to_encode = ["Funding Stage", "Industry"]
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_data = ohe.fit_transform(df[cols_to_encode])
encoded_df = pd.DataFrame(
    encoded_data, columns=ohe.get_feature_names_out(cols_to_encode)
)
df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

# Engineer tech stack features
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

# Drop original transformed columns
df = df.drop(
    columns=["Country", "Funding Stage", "Industry", "Tech Stack"], errors="ignore"
)

# Log Transformation
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
    if col in df.columns:
        df[col] = np.log1p(df[col])
        new_col_name = f"Log_{col}"
        df.rename(columns={col: new_col_name}, inplace=True)
        log_skewed_cols.append(new_col_name)

print("--- Feature Engineering Complete ---")
print("\n" + "=" * 50 + "\n")


# =============================================================================
# 4. SAVE DATASET WITH OUTLIERS
# =============================================================================
# At this point, `df` is fully processed but still contains outliers.
df_with_outliers = df.copy()

# Scale the features for the "with outliers" dataset
scaler_with_outliers = StandardScaler()
features_to_scale_1 = df_with_outliers.select_dtypes(include=np.number).columns.drop(
    ["StartupID", "Success Score"]
)
df_with_outliers[features_to_scale_1] = scaler_with_outliers.fit_transform(
    df_with_outliers[features_to_scale_1]
)

# Save to CSV
df_with_outliers.to_csv("data_with_outliers.csv", index=False)
print(
    f"Successfully saved 'data_with_outliers.csv' with shape {df_with_outliers.shape}"
)
print("\n" + "=" * 50 + "\n")


# =============================================================================
# 5. REMOVE OUTLIERS AND SAVE SECOND DATASET
# =============================================================================
print("--- Removing Outliers using IQR ---")
outlier_mask = pd.Series(False, index=df.index)
for col in log_skewed_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = outlier_mask | (df[col] < lower_bound) | (df[col] > upper_bound)

df_without_outliers = df[~outlier_mask].copy()
print(f"Removed {outlier_mask.sum()} outlier rows.")

# Scale the features for the "without outliers" dataset
if not df_without_outliers.empty:
    scaler_without_outliers = StandardScaler()
    features_to_scale_2 = df_without_outliers.select_dtypes(
        include=np.number
    ).columns.drop(["StartupID", "Success Score"])
    df_without_outliers[features_to_scale_2] = scaler_without_outliers.fit_transform(
        df_without_outliers[features_to_scale_2]
    )

    # Save to CSV
    df_without_outliers.to_csv("data_without_outliers.csv", index=False)
    print(
        f"Successfully saved 'data_without_outliers.csv' with shape {df_without_outliers.shape}"
    )
else:
    print(
        "Warning: DataFrame is empty after outlier removal. 'data_without_outliers.csv' was not saved."
    )

print("\n" + "=" * 50 + "\n")
print("Data preparation complete. You now have two CSV files ready for model training.")
