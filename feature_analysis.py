# -*- coding: utf-8 -*-
"""
Normality Check for Top Features in the Startup Success Dataset

This script identifies the top 15 features based on mutual information
and generates visualizations (Histograms and Q-Q Plots) to assess
if they are normally distributed.
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
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split


# =============================================================================
# 2. DATA LOADING AND PREPARATION (Condensed from previous script)
# =============================================================================
# This section quickly reproduces the cleaned DataFrame `df_cleaned`

# Load data
file_path = "global_startup_success_dataset.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "hamnakaleemds/global-startup-success-dataset",
    file_path,
)
df.drop_duplicates(inplace=True)

# Feature Engineering
df['StartupID'] = df['Startup Name'].str.extract(r'_(\d+)', expand=False).astype('Int64')
binary_map = {'Yes': 1, 'No': 0}
df['Acquired?'] = df['Acquired?'].map(binary_map)
df['IPO?'] = df['IPO?'].map(binary_map)
current_year = datetime.datetime.now().year
df['Startup Age'] = current_year - df['Founded Year']
df['Number of Employees'] = df['Number of Employees'].replace(0, np.nan)
df['Funding per Employee'] = df['Total Funding ($M)'] / df['Number of Employees']
df['Revenue per Employee'] = df['Annual Revenue ($M)'] / df['Number of Employees']
df[['Funding per Employee', 'Revenue per Employee']] = df[['Funding per Employee', 'Revenue per Employee']].fillna(0)
df.replace([np.inf, -np.inf], 0, inplace=True)
country_freq = df['Country'].value_counts(normalize=True)
df['Country_Encoded'] = df['Country'].map(country_freq)
cols_to_encode = ['Funding Stage', 'Industry']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = ohe.fit_transform(df[cols_to_encode])
encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(cols_to_encode))
df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
df['Uses_Python'] = df['Tech Stack'].str.contains('Python', case=False, na=False).astype(int)
df['Uses_Java'] = df['Tech Stack'].str.contains('Java', case=False, na=False).astype(int)
df['Uses_Nodejs'] = df['Tech Stack'].str.contains('Node.js', case=False, na=False).astype(int)
df['Uses_AI'] = df['Tech Stack'].str.contains('AI', case=False, na=False).astype(int)
df = df.drop(columns=['Startup Name', 'Country', 'Funding Stage', 'Industry', 'Tech Stack'], errors='ignore')

# Log Transformation
skewed_cols = ['Total Funding ($M)', 'Number of Employees', 'Annual Revenue ($M)',
               'Valuation ($B)', 'Customer Base (Millions)', 'Social Media Followers',
               'Funding per Employee', 'Revenue per Employee']
log_skewed_cols = []
for col in skewed_cols:
    if col in df.columns:
        df[col] = np.log1p(df[col])
        new_col_name = f'Log_{col}'
        df.rename(columns={col: new_col_name}, inplace=True)
        log_skewed_cols.append(new_col_name)

# Outlier Removal
outlier_mask = pd.Series(False, index=df.index)
for col in log_skewed_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = outlier_mask | (df[col] < lower_bound) | (df[col] > upper_bound)
df_cleaned = df[~outlier_mask].copy()

# Feature Scaling
if not df_cleaned.empty:
    scaler = StandardScaler()
    features_to_scale = df_cleaned.select_dtypes(include=np.number).columns.drop(['StartupID', 'Success Score'])
    df_cleaned[features_to_scale] = scaler.fit_transform(df_cleaned[features_to_scale])

print("Data preparation complete. Proceeding to normality check.")
print("\n" + "="*50 + "\n")


# =============================================================================
# 3. NORMALITY CHECK FOR TOP FEATURES
# =============================================================================
if not df_cleaned.empty:
    print("--- Checking for Normal Distribution in Top 15 Features ---")
    
    # Define features (X) and target (y)
    X = df_cleaned.drop(columns=['StartupID', 'Success Score'])
    y = df_cleaned['Success Score']

    # Identify top 15 features using Mutual Information
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    top_15_features = mi_scores.sort_values(ascending=False).head(15).index.tolist()
    
    print("Top 15 features to be analyzed:", top_15_features)
    print("\nGenerating plots...")

    # Create plots for each of the top 15 features
    for feature in top_15_features:
        plt.figure(figsize=(12, 5))
        
        # 1. Histogram with KDE
        plt.subplot(1, 2, 1)
        sns.histplot(df_cleaned[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
        
        # 2. Q-Q Plot
        plt.subplot(1, 2, 2)
        stats.probplot(df_cleaned[feature], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {feature}')
        
        plt.tight_layout()
        plt.show()

else:
    print("Cannot perform normality check because the dataframe is empty.")

print("\nNormality check complete.")