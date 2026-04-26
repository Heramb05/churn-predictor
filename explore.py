import pandas as pd

# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# See the first 5 rows
print("First 5 rows:")
print(df.head())

# See the shape (rows, columns)
print("\nDataset shape:", df.shape)

# See column names
print("\nColumns:", df.columns.tolist())

# How many customers churned vs stayed
print("\nChurn counts:")
print(df["Churn"].value_counts())