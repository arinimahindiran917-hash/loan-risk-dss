import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("models/confusion_matrix.png")
plt.show()