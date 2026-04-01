import pandas as pd


df = pd.read_csv("data/credit_default.csv")

print("✅ Dataset Loaded Successfully!")
print("Shape (rows, columns):", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())