import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

df = pd.read_csv("data/credit_default.csv")

df = df.iloc[1:]
df = df.apply(pd.to_numeric, errors="coerce")

#histogram
plt.figure(figsize=(12, 6))
sns.histplot(df["X1"], bins=30, kde=True, color="steelblue")

plt.title("Distribution of Credit Limit ($X_1$)", fontsize=18)
plt.xlabel("Credit Limit", fontsize=13)
plt.ylabel("Frequency", fontsize=13)

plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("Figure_4_1_Histogram_Final.png", dpi=300)
plt.show()

# Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["X1"], color="steelblue")

plt.title("Boxplot of Credit Limit ($X_1$)", fontsize=16)
plt.xlabel("Credit Limit", fontsize=12)

plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

plt.tight_layout()
plt.savefig("Figure_4_1_Boxplot_Final.png", dpi=300)
plt.show()