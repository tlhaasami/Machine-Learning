import pandas as pd
import os

# Load dataset
data_path = "DataSet/energy_efficiency_data.csv"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

df = pd.read_csv(data_path)

# Save the entire labeled dataset (all columns) to cleaned.csv
df.to_csv("cleaned.csv", index=False)

print("Data preparation complete:")
print(f"Full labeled dataset saved to: cleaned.csv (Shape: {df.shape})")
print("\nColumns:")
print(list(df.columns))
