# -*- coding: utf-8 -*-
"""
Data Cleansing and Exploration of Global Startup Success Dataset

This script loads the Global Startup Success Dataset, performs cleaning operations,
feature engineering, and conducts initial exploratory data analysis.
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression, SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# =============================================================================
# 2. INITIAL SETUP & DATA LOADING
# =============================================================================
# Set pandas display option to show all columns
pd.set_option('display.max_columns', None)

# Load the dataset using KaggleHub
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
print("\n" + "="*50 + "\n")

print("Dataset Information:")
df.info()
print("\n" + "="*50 + "\n")

# Check for null values
print("Null value counts per column:")
print(df.isna().sum())
print("\n" + "="*50 + "\n")

# Check for and remove duplicates
print(f"Number of rows before removing duplicates: {len(df)}")
df.drop_duplicates(inplace=True)
print(f"Number of rows after removing duplicates: {len(df)}")
print("\n" + "="*50 + "\n")


# =============================================================================
# 4. DATA CLEANING AND PREPROCESSING
# =============================================================================
print("--- Cleaning and Preprocessing ---")

# Convert 'Startup Name' to a numeric 'StartupID'
print("Converting 'Startup Name' to numeric 'StartupID'...")
df['StartupID'] = df['Startup Name'].str.extract(r'_(\d+)', expand=False).astype('Int64')
df = df.drop(columns=['Startup Name'])
# Move StartupID to the first column
cols = list(df.columns)
cols.remove('StartupID')
cols.insert(0, 'StartupID')
df = df[cols]
print("'StartupID' created and moved to the front.")
print("\n" + "="*30 + "\n")

# Convert binary categorical columns ('Acquired?', 'IPO?') to numeric (0/1)
print("Converting binary columns ('Acquired?', 'IPO?') to numeric...")
binary_map = {'Yes': 1, 'No': 0}
df['Acquired?'] = df['Acquired?'].map(binary_map)
df['IPO?'] = df['IPO?'].map(binary_map)
print("Binary columns converted.")
print("\n" + "="*30 + "\n")

print("Data types after initial cleaning:")
df.info()
print("\n" + "="*50 + "\n")


# =============================================================================
# 4.5 FEATURE ENGINEERING
# =============================================================================
print("--- Feature Engineering ---")

# 1. Create 'Startup Age'
print("Creating 'Startup Age' feature...")
current_year = datetime.datetime.now().year
df['Startup Age'] = current_year - df['Founded Year']
print("Done.")
print("\n" + "="*30 + "\n")

# 2. Engineer Financial Ratios
print("Creating financial ratio features...")
# Replace 0s in 'Number of Employees' with NaN to avoid division by zero errors, then fill resulting NaNs
df['Number of Employees'] = df['Number of Employees'].replace(0, np.nan)
df['Funding per Employee'] = df['Total Funding ($M)'] / df['Number of Employees']
df['Revenue per Employee'] = df['Annual Revenue ($M)'] / df['Number of Employees']
# Fill any potential NaN/inf values in the new ratio columns with 0
df[['Funding per Employee', 'Revenue per Employee']] = df[['Funding per Employee', 'Revenue per Employee']].fillna(0)
df.replace([np.inf, -np.inf], 0, inplace=True)
print("Done.")
print("\n" + "="*30 + "\n")

# 3. Advanced Encoding for Categorical Features
print("Applying advanced encoding to categorical features...")
# Frequency Encoding for 'Country'
country_freq = df['Country'].value_counts(normalize=True)
df['Country_Encoded'] = df['Country'].map(country_freq)

# One-Hot Encoding for 'Funding Stage'
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
funding_stage_encoded = ohe.fit_transform(df[['Funding Stage']])
funding_stage_df = pd.DataFrame(funding_stage_encoded, columns=ohe.get_feature_names_out(['Funding Stage']))
df = pd.concat([df.reset_index(drop=True), funding_stage_df], axis=1)

# Drop original columns that have been encoded
df = df.drop(columns=['Country', 'Funding Stage'])
print("Done.")
print("\n" + "="*30 + "\n")


# 4. Extract Information from 'Tech Stack'
print("Extracting features from 'Tech Stack'...")
df['Uses_Python'] = df['Tech Stack'].str.contains('Python', case=False, na=False).astype(int)
df['Uses_Java'] = df['Tech Stack'].str.contains('Java', case=False, na=False).astype(int)
df['Uses_Nodejs'] = df['Tech Stack'].str.contains('Node.js', case=False, na=False).astype(int)
df['Uses_AI'] = df['Tech Stack'].str.contains('AI', case=False, na=False).astype(int)

# Drop the original 'Tech Stack' column
df = df.drop(columns=['Tech Stack'])
print("Done.")
print("\n" + "="*30 + "\n")

print("DataFrame head after feature engineering:")
print(df.head())
print("\n" + "="*50 + "\n")


# =============================================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("--- Exploratory Data Analysis ---")

# Separate numeric and non-numeric columns for analysis
numeric_features = df.select_dtypes(include=np.number).columns.tolist()
non_numeric_features = df.select_dtypes(include='object').columns.tolist()

print("Numeric features identified:")
print(numeric_features)
print("\nNon-numeric features identified (should be few after encoding):")
print(non_numeric_features)
print("\n" + "="*30 + "\n")

# Analyze value counts for any remaining categorical features
if non_numeric_features:
    for col in non_numeric_features:
        print(f"--- Value Counts for '{col}' ---")
        print(df[col].value_counts().head(15))
        print()
else:
    print("No non-numeric features remain to be analyzed.")
print("\n" + "="*50 + "\n")

# Visualize distributions of key numerical features using boxplots
print("Generating boxplots for key numerical features...")
plt.figure(figsize=(18, 6))
plt.suptitle('Distributions of Key Numerical Features', fontsize=16)

# Plot for Total Funding
plt.subplot(1, 3, 1) # (1 row, 3 columns, 1st plot)
sns.boxplot(y=df['Total Funding ($M)'])
plt.title('Total Funding')

# Plot for Annual Revenue
plt.subplot(1, 3, 2) # (1 row, 3 columns, 2nd plot)
sns.boxplot(y=df['Annual Revenue ($M)'])
plt.title('Annual Revenue')

# Plot for Number of Employees
plt.subplot(1, 3, 3) # (1 row, 3 columns, 3rd plot)
sns.boxplot(y=df['Number of Employees'])
plt.title('Number of Employees')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
print("Script execution complete.")
