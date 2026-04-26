import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle

# ── Load & Clean ──────────────────────────────────────────
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df.drop("customerID", axis=1, inplace=True)

# ── Encode categorical columns ────────────────────────────
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# ── Split features and target ─────────────────────────────
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ── Train/Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Scale numeric features ────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
np.save("X_train_scaled.npy", X_train_scaled)

# ── Train 3 models & compare ──────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

best_auc = 0
best_model = None
best_name = ""

print("=" * 50)
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    print(f"\n📊 {name}")
    print(f"   AUC-ROC Score: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_name = name

print("=" * 50)
print(f"\n🏆 Best Model: {best_name} with AUC = {best_auc:.4f}")

# ── Save best model & scaler ──────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\n✅ Model, scaler and columns saved!")