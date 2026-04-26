import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
X_train_scaled = np.load("X_train_scaled.npy")

# ── Load model, scaler, columns ───────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

# ── Page config ───────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.write("Fill in the customer details below to predict if they will churn.")

# ── Input form ────────────────────────────────────────────
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.slider("Monthly Charges ($)", 0.0, 120.0, 65.0)
    total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)

# ── Encode inputs same way as training ────────────────────
def encode(val, mapping):
    return mapping.get(val, 0)

yes_no = {"No": 0, "Yes": 1}
yes_no_nps = {"No": 0, "Yes": 1, "No phone service": 2}
yes_no_nis = {"No": 0, "Yes": 1, "No internet service": 2}

input_data = {
    "gender": encode(gender, {"Female": 0, "Male": 1}),
    "SeniorCitizen": encode(senior, yes_no),
    "Partner": encode(partner, yes_no),
    "Dependents": encode(dependents, yes_no),
    "tenure": tenure,
    "PhoneService": encode(phone_service, yes_no),
    "MultipleLines": encode(multiple_lines, yes_no_nps),
    "InternetService": encode(internet_service, {"DSL": 0, "Fiber optic": 1, "No": 2}),
    "OnlineSecurity": encode(online_security, yes_no_nis),
    "OnlineBackup": encode(online_backup, yes_no_nis),
    "DeviceProtection": encode(device_protection, yes_no_nis),
    "TechSupport": encode(tech_support, yes_no_nis),
    "StreamingTV": encode(streaming_tv, yes_no_nis),
    "StreamingMovies": encode(streaming_movies, yes_no_nis),
    "Contract": encode(contract, {"Month-to-month": 0, "One year": 1, "Two year": 2}),
    "PaperlessBilling": encode(paperless_billing, yes_no),
    "PaymentMethod": encode(payment_method, {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }),
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

input_df = pd.DataFrame([input_data])[columns]
input_scaled = scaler.transform(input_df)

# ── Predict ───────────────────────────────────────────────
if st.button("🔮 Predict Churn"):
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = "Will Churn ⚠️" if prob >= 0.5 else "Will NOT Churn ✅"

    st.divider()
    st.subheader("Prediction Result")

    if prob >= 0.5:
        st.error(f"**{prediction}**")
    else:
        st.success(f"**{prediction}**")

    st.metric("Churn Probability", f"{prob*100:.1f}%")
    st.progress(float(prob))

    st.divider()
    # ── SHAP Explanation ──────────────────────────────────
    st.subheader("🔍 Why this prediction?")
    explainer = shap.LinearExplainer(model, X_train_scaled if hasattr(model, 'coef_') else input_scaled)
    shap_values = explainer.shap_values(input_scaled)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_scaled[0],
            feature_names=columns
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()
    st.caption("Built with ❤️ using Scikit-learn + Streamlit")