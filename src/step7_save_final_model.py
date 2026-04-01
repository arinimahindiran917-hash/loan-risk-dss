import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("data/credit_default.csv")

df = df[df["X1"].astype(str).str.isnumeric()].copy()

for col in df.columns:
    if col != "Unnamed: 0":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop(columns=["Unnamed: 0"])

X = df.drop("Y", axis=1)
y = df["Y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "models/final_model.pkl")

print("Final model saved successfully!")