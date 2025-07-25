import pandas as pd
import numpy as np

# Set display options
pd.set_option("display.max_columns", None)

# Load the CSV file (no need for kagglehub anymore since you downloaded it with the CLI)
print("Loading dataset...")
file_path = "data/startup_data.csv"  # Correct relative path and filename
df = pd.read_csv(file_path)
print("Dataset loaded successfully.")
print(f"Initial shape: {df.shape}")
print("-" * 50)

# Preview data
print("First 5 rows:")
print(df.head())

print("\nInfo:")
df.info()
