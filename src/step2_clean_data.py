import pandas as pd

df = pd.read_csv("data/credit_default.csv")

df = df[df["X1"].astype(str).str.isnumeric()].copy()

for col in df.columns:
    if col != "Unnamed: 0":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop(columns=["Unnamed: 0"])

print("✅ Cleaned dataset!")
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:", df.isnull().sum().sum())