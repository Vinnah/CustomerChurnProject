# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 🌟 Load the saved model
model = joblib.load("best_model.pkl")

# 🧮 Expected features (based on your model input)
feature_columns = [
    "monthly_spend", "total_transaction_value", "months_active",
    "avg_spend_per_month", "gender", "SeniorCitizen", "Partner",
    "Dependents", "has_cash_card", "MultipleLines", "digital_access_level",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "subscription_type", "PaperlessBilling",
    "PaymentMethod"
]

# 🧾 App Title
st.title("💳 Fintech Churn Predictor")

# 📥 Input Option: Upload CSV or Manual
upload = st.file_uploader("Upload customer data (CSV format)", type=["csv"])

if upload:
    df = pd.read_csv(upload)
else:
    st.write("Or enter values manually:")
    with st.form("churn_form"):
        monthly_spend = st.number_input("Monthly Spend (ZAR)", min_value=0.0, step=1.0)
        total_transaction_value = st.number_input("Total Transaction Value (ZAR)", min_value=0.0, step=1.0)
        months_active = st.number_input("Months Active", min_value=0, step=1)
        avg_spend_per_month = total_transaction_value / months_active if months_active > 0 else 0

        gender = st.selectbox("Gender", [0, 1])  # 0 = Female, 1 = Male
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Has Partner", [0, 1])
        Dependents = st.selectbox("Has Dependents", [0, 1])
        has_cash_card = st.selectbox("Has Cash Card", [0, 1])
        MultipleLines = st.selectbox("Multiple Lines", [0, 1])
        digital_access_level = st.selectbox("Digital Access Level", [0, 1, 2])
        OnlineSecurity = st.selectbox("Online Security", [0, 1, 2])
        OnlineBackup = st.selectbox("Online Backup", [0, 1, 2])
        DeviceProtection = st.selectbox("Device Protection", [0, 1, 2])
        TechSupport = st.selectbox("Tech Support", [0, 1, 2])
        StreamingTV = st.selectbox("Streaming TV", [0, 1, 2])
        StreamingMovies = st.selectbox("Streaming Movies", [0, 1, 2])
        subscription_type = st.selectbox("Subscription Type", [0, 1, 2])
        PaperlessBilling = st.selectbox("Paperless Billing", [0, 1])
        PaymentMethod = st.selectbox("Payment Method", [0, 1, 2, 3])

        submit = st.form_submit_button("Predict")

    if submit:
        input_data = pd.DataFrame([[
            monthly_spend, total_transaction_value, months_active, avg_spend_per_month,
            gender, SeniorCitizen, Partner, Dependents, has_cash_card, MultipleLines,
            digital_access_level, OnlineSecurity, OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV, StreamingMovies, subscription_type,
            PaperlessBilling, PaymentMethod
        ]], columns=feature_columns)

        churn_pred = model.predict(input_data)[0]
        churn_prob = model.predict_proba(input_data)[0][1]

        st.success(f"Prediction: {'🔴 Churn Risk' if churn_pred == 1 else '🟢 Not Likely to Churn'}")
        st.info(f"Churn Probability: {churn_prob:.2%}")

# 📊 Optional: Show example format
with st.expander("ℹ️ View example data format"):
    st.write(pd.DataFrame(columns=feature_columns).head(1))
