import pandas as pd

df = pd.read_csv("data/credit_default.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df.apply(pd.to_numeric, errors="coerce")

columns = ["X1","X5","X6","X7","X12","X13","X18","X19","Y"]

stats = df[columns].describe().T

stats = stats.loc[:, ["mean","std","min","25%","50%","75%","max"]]

stats.columns = ["Mean","Std Dev","Min","25%","Median","75%","Max"]

stats.index = [
    "Credit Limit (X1)",
    "Age (X5)",
    "Repayment Status Sep (X6)",
    "Repayment Status Aug (X7)",
    "Bill Amount Sep (X12)",
    "Bill Amount Aug (X13)",
    "Payment Amount Sep (X18)",
    "Payment Amount Aug (X19)",
    "Default Target (Y)"
]

stats = stats.round(2)

print("\nTable 4.1 Output:\n")
print(stats)

stats.to_excel("Table_4_1_Descriptive_Statistics.xlsx")

print("\nSaved successfully.")