import pandas as pd
import numpy as np
import os
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

import kagglehub
from kagglehub import KaggleDatasetAdapter
# display all cols
pd.set_option('display.max_columns', None)
# Set the path to the file you'd like to load
file_path = "global_startup_success_dataset.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "hamnakaleemds/global-startup-success-dataset",
  file_path,
  # Provide any additional arguments like
  # sql_query or pandas_kwargs. See the
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())
print("Number of rows in dataset:", len(df))

df.info()

# null counts
print(df.isna().sum())

# no null, probably already been cleaned

df_with_numeric_features = df.select_dtypes(exclude=['object'])
print("Numeric features:")
for col in df_with_numeric_features:
    print(f"{col}")