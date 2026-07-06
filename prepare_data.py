import pandas as pd
import os

# Load dataset
data_path = "DataSet/energy_efficiency_data.csv"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

df = pd.read_csv(data_path)

# Separate features (first 8 columns)
# The dataset has exactly 10 columns: 8 features and 2 targets (Heating_Load, Cooling_Load)
features = df.iloc[:, :8]

# Save to CSV file
features.to_csv("cleaned.csv", index=False)

print("Data preparation complete:")
print(f"Features saved to: cleaned.csv (Shape: {features.shape})")
print("\nFeature Columns:")
print(list(features.columns))
