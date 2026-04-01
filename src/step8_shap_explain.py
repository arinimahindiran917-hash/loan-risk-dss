import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

model = joblib.load("models/final_model.pkl")

explainer = shap.Explainer(model, X_train)

shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("models/shap_summary.png", dpi=300, bbox_inches="tight")

print("SHAP summary plot saved successfully!")