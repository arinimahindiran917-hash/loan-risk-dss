import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


df = pd.read_csv("data/credit_default.csv")

df = df[df["X1"].astype(str).str.isnumeric()].copy()

for col in df.columns:
    if col != "Unnamed: 0":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop(columns=["Unnamed: 0"])

X = df.drop("Y", axis=1)
y = df["Y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))