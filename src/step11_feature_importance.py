import pandas as pd
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("data/credit_default.csv")

df = df[df["X1"].astype(str).str.isnumeric()].copy()

for col in df.columns:
    if col != "Unnamed: 0":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop(columns=["Unnamed: 0"])

X = df.drop("Y", axis=1)

model = joblib.load("models/final_model.pkl")

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.savefig("models/feature_importance.png")
plt.show()