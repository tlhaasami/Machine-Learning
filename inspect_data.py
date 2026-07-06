import pandas as pd
import os

data_path = "DataSet/energy_efficiency_data.csv"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

df = pd.read_csv(data_path)
print("--- Data Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Duplicate Rows ---")
print(f"Number of duplicate rows: {df.duplicated().sum()}")
print("\n--- Descriptive Statistics ---")
print(df.describe())
print("\n--- Unique Values in Categorical/Discrete Columns ---")
for col in ['Overall_Height', 'Orientation', 'Glazing_Area_Distribution']:
    if col in df.columns:
        print(f"{col}: {df[col].unique()}")
