# # -*- coding: utf-8 -*-
# """
# Data Cleansing and Exploration of Global Startup Success Dataset

# This script loads the Global Startup Success Dataset, performs cleaning operations,
# feature engineering, and conducts initial exploratory data analysis.
# """

# # =============================================================================
# # 1. IMPORT LIBRARIES
# # =============================================================================
# import pandas as pd
# import numpy as np
# import os
# import datetime
# from scipy.stats import zscore
# import matplotlib.pyplot as plt
# import seaborn as sns
# import kagglehub
# from kagglehub import KaggleDatasetAdapter
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
# from sklearn.feature_selection import mutual_info_regression, SequentialFeatureSelector
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import RandomForestRegressor
# from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# # =============================================================================
# # 2. INITIAL SETUP & DATA LOADING
# # =============================================================================
# # Set pandas display option to show all columns
# pd.set_option('display.max_columns', None)

# # Load the dataset using KaggleHub
# print("Loading dataset...")
# file_path = "global_startup_success_dataset.csv"
# df = kagglehub.load_dataset(
#     KaggleDatasetAdapter.PANDAS,
#     "hamnakaleemds/global-startup-success-dataset",
#     file_path,
# )
# print("Dataset loaded successfully.")
# print("-" * 50)


# # =============================================================================
# # 3. INITIAL DATA INSPECTION
# # =============================================================================
# print("First 5 records of the dataset:")
# print(df.head())
# print("\n" + "="*50 + "\n")

# print("Dataset Information:")
# df.info()
# print("\n" + "="*50 + "\n")

# # Check for null values
# print("Null value counts per column:")
# print(df.isna().sum())
# print("\n" + "="*50 + "\n")

# # Check for and remove duplicates
# print(f"Number of rows before removing duplicates: {len(df)}")
# df.drop_duplicates(inplace=True)
# print(f"Number of rows after removing duplicates: {len(df)}")
# print("\n" + "="*50 + "\n")


# # =============================================================================
# # 4. DATA CLEANING AND PREPROCESSING
# # =============================================================================
# print("--- Cleaning and Preprocessing ---")

# # Convert 'Startup Name' to a numeric 'StartupID'
# print("Converting 'Startup Name' to numeric 'StartupID'...")
# df['StartupID'] = df['Startup Name'].str.extract(r'_(\d+)', expand=False).astype('Int64')
# df = df.drop(columns=['Startup Name'])
# # Move StartupID to the first column
# cols = list(df.columns)
# cols.remove('StartupID')
# cols.insert(0, 'StartupID')
# df = df[cols]
# print("'StartupID' created and moved to the front.")
# print("\n" + "="*30 + "\n")

# # Convert binary categorical columns ('Acquired?', 'IPO?') to numeric (0/1)
# print("Converting binary columns ('Acquired?', 'IPO?') to numeric...")
# binary_map = {'Yes': 1, 'No': 0}
# df['Acquired?'] = df['Acquired?'].map(binary_map)
# df['IPO?'] = df['IPO?'].map(binary_map)
# print("Binary columns converted.")
# print("\n" + "="*30 + "\n")

# print("Data types after initial cleaning:")
# df.info()
# print("\n" + "="*50 + "\n")


# # =============================================================================
# # 4.5 FEATURE ENGINEERING
# # =============================================================================
# # print("--- Feature Engineering ---")

# # # 1. Create 'Startup Age'
# # print("Creating 'Startup Age' feature...")
# # current_year = datetime.datetime.now().year
# # df['Startup Age'] = current_year - df['Founded Year']
# # print("Done.")
# # print("\n" + "="*30 + "\n")

# # # 2. Engineer Financial Ratios
# # print("Creating financial ratio features...")
# # # Replace 0s in 'Number of Employees' with NaN to avoid division by zero errors, then fill resulting NaNs
# # df['Number of Employees'] = df['Number of Employees'].replace(0, np.nan)
# # df['Funding per Employee'] = df['Total Funding ($M)'] / df['Number of Employees']
# # df['Revenue per Employee'] = df['Annual Revenue ($M)'] / df['Number of Employees']
# # # Fill any potential NaN/inf values in the new ratio columns with 0
# # df[['Funding per Employee', 'Revenue per Employee']] = df[['Funding per Employee', 'Revenue per Employee']].fillna(0)
# # df.replace([np.inf, -np.inf], 0, inplace=True)
# # print("Done.")
# # print("\n" + "="*30 + "\n")

# # # 3. Advanced Encoding for Categorical Features
# # print("Applying advanced encoding to categorical features...")
# # # Frequency Encoding for 'Country'
# # country_freq = df['Country'].value_counts(normalize=True)
# # df['Country_Encoded'] = df['Country'].map(country_freq)

# # # One-Hot Encoding for 'Funding Stage'
# # ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# # funding_stage_encoded = ohe.fit_transform(df[['Funding Stage']])
# # funding_stage_df = pd.DataFrame(funding_stage_encoded, columns=ohe.get_feature_names_out(['Funding Stage']))
# # df = pd.concat([df.reset_index(drop=True), funding_stage_df], axis=1)

# # # Drop original columns that have been encoded
# # df = df.drop(columns=['Country', 'Funding Stage'])
# # print("Done.")
# # print("\n" + "="*30 + "\n")


# # # 4. Extract Information from 'Tech Stack'
# # print("Extracting features from 'Tech Stack'...")
# # df['Uses_Python'] = df['Tech Stack'].str.contains('Python', case=False, na=False).astype(int)
# # df['Uses_Java'] = df['Tech Stack'].str.contains('Java', case=False, na=False).astype(int)
# # df['Uses_Nodejs'] = df['Tech Stack'].str.contains('Node.js', case=False, na=False).astype(int)
# # df['Uses_AI'] = df['Tech Stack'].str.contains('AI', case=False, na=False).astype(int)

# # # Drop the original 'Tech Stack' column
# # df = df.drop(columns=['Tech Stack'])
# # print("Done.")
# # print("\n" + "="*30 + "\n")

# # print("DataFrame head after feature engineering:")
# # print(df.head())
# # print("\n" + "="*50 + "\n")

# # =============================
# # CHANGING FEATURE ENGINEERING 
# # +++++++++++++++++++++++++++++

# print("--- CHANGED Feature Engineering ---")

# # 1. Create 'Startup Age'
# print("Creating 'Startup Age' feature...")
# current_year = datetime.datetime.now().year
# df['Startup Age'] = current_year - df['Founded Year']
# print("Done.")
# print("\n" + "="*30 + "\n")

# # 2. Engineer Financial Ratios
# print("Creating financial ratio features...")
# # Replace 0s in 'Number of Employees' with NaN to avoid division by zero errors, then fill resulting NaNs
# df['Number of Employees'] = df['Number of Employees'].replace(0, 1)

# df['Funding per Employee'] = df['Total Funding ($M)'] / df['Number of Employees']
# df['Revenue per Employee'] = df['Annual Revenue ($M)'] / df['Number of Employees']

# df['Valuation_to_Revenue'] = df['Valuation ($B)'] * 1000 / df['Annual Revenue ($M)']
# df['Valuation_to_Funding'] = df['Valuation ($B)'] * 1000 / df['Total Funding ($M)']
# df['Revenue_to_Funding'] = df['Annual Revenue ($M)'] / df['Total Funding ($M)']


# df['Log_Revenue'] = np.log1p(df['Annual Revenue ($M)'])
# df['Log_Valuation'] = np.log1p(df['Valuation ($B)'])
# df['Log_Total_Funding'] = np.log1p(df['Total Funding ($M)'])
# df['Log_Employees'] = np.log1p(df['Number of Employees'])
# df['Log_Customer_Base'] = np.log1p(df['Customer Base (Millions)'])

# # Customer efficiency metrics
# df['Revenue_per_Customer'] = df['Annual Revenue ($M)'] / df['Customer Base (Millions)']
# df['Employees_per_Customer'] = df['Number of Employees'] / df['Customer Base (Millions)']
# df['Social_Media_per_Customer'] = df['Social Media Followers'] / df['Customer Base (Millions)']

# # Growth and efficiency indicators
# df['Funding_per_Year'] = df['Total Funding ($M)'] / df['Startup Age']
# df['Revenue_per_Year'] = df['Annual Revenue ($M)'] / df['Startup Age']
# df['Customer_Growth_Rate'] = df['Customer Base (Millions)'] / df['Startup Age']

# # Replace infinite values with NaN, then fill with median
# df.replace([np.inf, -np.inf], np.nan, inplace=True)

# # Fill NaN values in ratio columns with median
# ratio_columns = ['Funding per Employee', 'Revenue per Employee', 'Valuation_to_Revenue', 
#                 'Valuation_to_Funding', 'Revenue_to_Funding', 'Revenue_per_Customer',
#                 'Employees_per_Customer', 'Social_Media_per_Customer', 'Funding_per_Year',
#                 'Revenue_per_Year', 'Customer_Growth_Rate']

# for col in ratio_columns:
#     if col in df.columns:
#         df[col] = df[col].fillna(df[col].median())

# print("Financial features created.")
# print("\n" + "="*30 + "\n")

# # 3. Advanced Encoding for Categorical Features
# print("Applying advanced encoding to categorical features...")
# # Frequency Encoding for 'Country'
# country_freq = df['Country'].value_counts(normalize=True)
# df['Country_Encoded'] = df['Country'].map(country_freq)

# # One-Hot Encoding for 'Funding Stage'
# ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# funding_stage_encoded = ohe.fit_transform(df[['Funding Stage']])
# funding_stage_df = pd.DataFrame(funding_stage_encoded, columns=ohe.get_feature_names_out(['Funding Stage']))
# df = pd.concat([df.reset_index(drop=True), funding_stage_df], axis=1)

# # Drop original columns that have been encoded
# df = df.drop(columns=['Country', 'Funding Stage'])
# print("Done.")
# print("\n" + "="*30 + "\n")


# # 4. Extract Information from 'Tech Stack'
# print("Extracting features from 'Tech Stack'...")
# df['Uses_Python'] = df['Tech Stack'].str.contains('Python', case=False, na=False).astype(int)
# df['Uses_Java'] = df['Tech Stack'].str.contains('Java', case=False, na=False).astype(int)
# df['Uses_Nodejs'] = df['Tech Stack'].str.contains('Node.js', case=False, na=False).astype(int)
# df['Uses_AI'] = df['Tech Stack'].str.contains('AI', case=False, na=False).astype(int)

# # Drop the original 'Tech Stack' column
# df = df.drop(columns=['Tech Stack'])
# print("Done.")
# print("\n" + "="*30 + "\n")

# print("DataFrame head after feature engineering:")
# print(df.head())
# print("\n" + "="*50 + "\n")

# # =============================================================================
# # 5. EXPLORATORY DATA ANALYSIS (EDA)
# # =============================================================================
# print("--- Exploratory Data Analysis ---")

# # Separate numeric and non-numeric columns for analysis
# numeric_features = df.select_dtypes(include=np.number).columns.tolist()
# non_numeric_features = df.select_dtypes(include='object').columns.tolist()

# print("Numeric features identified:")
# print(numeric_features)
# print("\nNon-numeric features identified (should be few after encoding):")
# print(non_numeric_features)
# print("\n" + "="*30 + "\n")

# # Analyze value counts for any remaining categorical features
# if non_numeric_features:
#     for col in non_numeric_features:
#         print(f"--- Value Counts for '{col}' ---")
#         print(df[col].value_counts().head(15))
#         print()
# else:
#     print("No non-numeric features remain to be analyzed.")
# print("\n" + "="*50 + "\n")

# # Visualize distributions of key numerical features using boxplots
# print("Generating boxplots for key numerical features...")
# plt.figure(figsize=(18, 6))
# plt.suptitle('Distributions of Key Numerical Features', fontsize=16)

# # Plot for Total Funding
# plt.subplot(1, 3, 1) # (1 row, 3 columns, 1st plot)
# sns.boxplot(y=df['Total Funding ($M)'])
# plt.title('Total Funding')

# # Plot for Annual Revenue
# plt.subplot(1, 3, 2) # (1 row, 3 columns, 2nd plot)
# sns.boxplot(y=df['Annual Revenue ($M)'])
# plt.title('Annual Revenue')

# # Plot for Number of Employees
# plt.subplot(1, 3, 3) # (1 row, 3 columns, 3rd plot)
# sns.boxplot(y=df['Number of Employees'])
# plt.title('Number of Employees')

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
# print("Script execution complete.")

# -*- coding: utf-8 -*-
"""
Enhanced Data Cleansing and Feature Engineering for Global Startup Success Dataset
This script includes advanced feature engineering specifically designed to improve
model performance for predicting startup success scores.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
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

# =============================================================================
# 5. ENHANCED FEATURE ENGINEERING
# =============================================================================
print("--- Enhanced Feature Engineering ---")

# 1. Create 'Startup Age'
print("Creating 'Startup Age' feature...")
current_year = datetime.datetime.now().year
df['Startup Age'] = current_year - df['Founded Year']
print("Done.")
print("\n" + "="*30 + "\n")

# 2. Advanced Financial Features
print("Creating comprehensive financial features...")

# Handle zero employees issue more carefully
df['Number of Employees'] = df['Number of Employees'].replace(0, 1)  # Replace 0 with 1 to avoid division issues

# Basic ratios
df['Funding per Employee'] = df['Total Funding ($M)'] / df['Number of Employees']
df['Revenue per Employee'] = df['Annual Revenue ($M)'] / df['Number of Employees']

# Advanced financial ratios
df['Valuation_to_Revenue'] = df['Valuation ($B)'] * 1000 / df['Annual Revenue ($M)']
df['Valuation_to_Funding'] = df['Valuation ($B)'] * 1000 / df['Total Funding ($M)']
df['Revenue_to_Funding'] = df['Annual Revenue ($M)'] / df['Total Funding ($M)']

# Log transformations for skewed financial data
df['Log_Revenue'] = np.log1p(df['Annual Revenue ($M)'])
df['Log_Valuation'] = np.log1p(df['Valuation ($B)'])
df['Log_Total_Funding'] = np.log1p(df['Total Funding ($M)'])
df['Log_Employees'] = np.log1p(df['Number of Employees'])
df['Log_Customer_Base'] = np.log1p(df['Customer Base (Millions)'])

# Customer efficiency metrics
df['Revenue_per_Customer'] = df['Annual Revenue ($M)'] / df['Customer Base (Millions)']
df['Employees_per_Customer'] = df['Number of Employees'] / df['Customer Base (Millions)']
df['Social_Media_per_Customer'] = df['Social Media Followers'] / df['Customer Base (Millions)']

# Growth and efficiency indicators
df['Funding_per_Year'] = df['Total Funding ($M)'] / df['Startup Age']
df['Revenue_per_Year'] = df['Annual Revenue ($M)'] / df['Startup Age']
df['Customer_Growth_Rate'] = df['Customer Base (Millions)'] / df['Startup Age']

# Replace infinite values with NaN, then fill with median
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values in ratio columns with median
ratio_columns = ['Funding per Employee', 'Revenue per Employee', 'Valuation_to_Revenue', 
                'Valuation_to_Funding', 'Revenue_to_Funding', 'Revenue_per_Customer',
                'Employees_per_Customer', 'Social_Media_per_Customer', 'Funding_per_Year',
                'Revenue_per_Year', 'Customer_Growth_Rate']

for col in ratio_columns:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

print("Financial features created.")
print("\n" + "="*30 + "\n")

# 3. Business Model and Success Indicators
print("Creating business model indicators...")

# Success thresholds based on typical startup metrics
df['High_Revenue'] = (df['Annual Revenue ($M)'] > df['Annual Revenue ($M)'].quantile(0.75)).astype(int)
df['High_Valuation'] = (df['Valuation ($B)'] > df['Valuation ($B)'].quantile(0.75)).astype(int)
df['Unicorn'] = (df['Valuation ($B)'] >= 1.0).astype(int)
df['Large_Scale'] = (df['Number of Employees'] >= 1000).astype(int)
df['Mature_Company'] = (df['Startup Age'] >= 10).astype(int)
df['Young_Company'] = (df['Startup Age'] <= 5).astype(int)

# Efficiency indicators
df['High_Revenue_Efficiency'] = (df['Revenue per Employee'] > df['Revenue per Employee'].quantile(0.75)).astype(int)
df['High_Customer_Base'] = (df['Customer Base (Millions)'] > df['Customer Base (Millions)'].quantile(0.75)).astype(int)
df['High_Social_Media'] = (df['Social Media Followers'] > df['Social Media Followers'].quantile(0.75)).astype(int)

# Exit events
df['Exit_Event'] = ((df['Acquired?'] == 1) | (df['IPO?'] == 1)).astype(int)

print("Business indicators created.")
print("\n" + "="*30 + "\n")

# 4. Advanced Encoding for Categorical Features
print("Applying advanced encoding to categorical features...")

# Frequency Encoding for 'Country'
country_freq = df['Country'].value_counts(normalize=True)
df['Country_Encoded'] = df['Country'].map(country_freq)

# Target Encoding for Industry (more sophisticated than one-hot)
industry_success_mean = df.groupby('Industry')['Success Score'].mean()
df['Industry_Success_Mean'] = df['Industry'].map(industry_success_mean)

# One-Hot Encoding for 'Funding Stage'
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
funding_stage_encoded = ohe.fit_transform(df[['Funding Stage']])
funding_stage_df = pd.DataFrame(funding_stage_encoded, columns=ohe.get_feature_names_out(['Funding Stage']))
df = pd.concat([df.reset_index(drop=True), funding_stage_df], axis=1)

# Keep Industry as categorical for now (we'll create dummies later if needed)
print("Advanced encoding completed.")
print("\n" + "="*30 + "\n")

# 5. Extract Information from 'Tech Stack'
print("Extracting features from 'Tech Stack'...")
df['Uses_Python'] = df['Tech Stack'].str.contains('Python', case=False, na=False).astype(int)
df['Uses_Java'] = df['Tech Stack'].str.contains('Java', case=False, na=False).astype(int)
df['Uses_Nodejs'] = df['Tech Stack'].str.contains('Node.js', case=False, na=False).astype(int)
df['Uses_AI'] = df['Tech Stack'].str.contains('AI', case=False, na=False).astype(int)

# Count number of technologies used
tech_columns = ['Uses_Python', 'Uses_Java', 'Uses_Nodejs', 'Uses_AI']
df['Tech_Diversity'] = df[tech_columns].sum(axis=1)

# Drop original columns that have been encoded
df = df.drop(columns=['Country', 'Funding Stage', 'Tech Stack'])
print("Tech stack features extracted.")
print("\n" + "="*30 + "\n")

# =============================================================================
# 6. CORRELATION ANALYSIS BEFORE MODELING
# =============================================================================
print("--- Correlation Analysis ---")

# Calculate correlations with Success Score
numeric_features = df.select_dtypes(include=np.number).columns.tolist()
if 'Success Score' in numeric_features:
    numeric_features.remove('Success Score')
if 'StartupID' in numeric_features:
    numeric_features.remove('StartupID')

correlations = df[numeric_features + ['Success Score']].corr()['Success Score'].abs().sort_values(ascending=False)
print("Top 15 features by correlation with Success Score:")
print(correlations.head(15))
print("\n" + "="*30 + "\n")

# =============================================================================
# 7. PREPARE FEATURE MATRIX FOR MODELING
# =============================================================================
print("--- Preparing Feature Matrix ---")

def prepare_feature_matrix(df):
    """Prepare final feature matrix for modeling"""
    
    # Select all numeric features except target and ID
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    if 'Success Score' in numeric_features:
        numeric_features.remove('Success Score')
    if 'StartupID' in numeric_features:
        numeric_features.remove('StartupID')
    
    # Add Industry dummies if Industry column still exists
    if 'Industry' in df.columns:
        industry_dummies = pd.get_dummies(df['Industry'], prefix='Industry')
        feature_matrix = pd.concat([df[numeric_features], industry_dummies], axis=1)
    else:
        feature_matrix = df[numeric_features]
    
    return feature_matrix

X = prepare_feature_matrix(df)
y = df['Success Score']

print(f"Feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print(f"Features with highest correlation to target:")
correlations_final = X.corrwith(y).abs().sort_values(ascending=False)
print(correlations_final.head(10))
print("\n" + "="*30 + "\n")

# =============================================================================
# 8. MODEL PERFORMANCE COMPARISON: XGBoost, LightGBM, Linear Regression
# =============================================================================
print("--- Model Performance Comparison ---")

# Optional: install if needed
# !pip install xgboost lightgbm

import xgboost as xgb
import lightgbm as lgb

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    "Linear Regression": LinearRegression()
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{name} Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

    # Feature importance (only for tree-based models)
    if name in ["XGBoost", "LightGBM"]:
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
