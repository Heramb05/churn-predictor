import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Fix TotalCharges column (it has some empty strings)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# ── Chart 1: Churn Distribution ──────────────────────────
plt.figure(figsize=(6, 4))
df["Churn"].value_counts().plot(kind="bar", color=["steelblue", "salmon"])
plt.title("Churn vs No Churn")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("chart1_churn_distribution.png")
plt.show()
print("Chart 1 saved!")

# ── Chart 2: Churn by Contract Type ──────────────────────
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Contract", hue="Churn", palette=["steelblue", "salmon"])
plt.title("Churn by Contract Type")
plt.tight_layout()
plt.savefig("chart2_churn_by_contract.png")
plt.show()
print("Chart 2 saved!")

# ── Chart 3: Monthly Charges vs Churn ────────────────────
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette=["steelblue", "salmon"])
plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.savefig("chart3_monthly_charges.png")
plt.show()
print("Chart 3 saved!")

# ── Chart 4: Tenure vs Churn ──────────────────────────────
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, palette=["steelblue", "salmon"])
plt.title("Tenure vs Churn")
plt.tight_layout()
plt.savefig("chart4_tenure.png")
plt.show()
print("Chart 4 saved!")

print("\nAll charts saved to your folder!")