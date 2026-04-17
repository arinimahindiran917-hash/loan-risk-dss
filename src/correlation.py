import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/credit_default.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df.iloc[1:]

df = df.apply(pd.to_numeric, errors="coerce")

cols = ["X1", "X5", "X6", "X7", "X12", "X13", "X18", "X19", "Y"]

# Correlation matrix
corr = df[cols].corr()

# Plot
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Correlation Matrix of Selected Variables")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=300)
plt.show()