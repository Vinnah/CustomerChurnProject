# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load("best_model.pkl")

# Expected features (based on your model input)
feature_columns = [
    "monthly_spend", "total_transaction_value", "months_active",
    "avg_spend_per_month", "gender", "SeniorCitizen", "Partner",
    "Dependents", "has_cash_card", "MultipleLines", "digital_access_level",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "subscription_type", "PaperlessBilling",
    "PaymentMethod"
]

# App Title
st.title("Fintech Churn Predictor")

# Input Option: Upload CSV or Manual
upload = st.file_uploader("Upload customer data (CSV format)", type=["csv"])

if upload:
    df = pd.read_csv(upload)
else:
    st.write("Or enter values manually:")
    with st.form("churn_form"):

        st.markdown("### Input Customer Info")

        # Dropdown mapping
        def add_placeholder(options):
            return ["-- Select --"] + list(options)
    
        # Maps
        gender_map = {"Female": 0, "Male": 1}
        senior_map = {"No": 0, "Yes": 1}
        binary_map = {"No": 0, "Yes": 1}
        internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}
        service_map = {"No": 0, "Yes": 1, "No internet service": 2}
        streaming_map = {"No": 0, "Yes": 1, "No internet service": 2}
        subscription_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        payment_map = {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer (automatic)": 2,
            "Credit card (automatic)": 3
        }
    
        # Inputs
        gender_label = st.selectbox("Gender*", add_placeholder(gender_map.keys()))
        senior_label = st.selectbox("Senior Citizen*", add_placeholder(senior_map.keys()))
        partner_label = st.selectbox("Partner*", add_placeholder(binary_map.keys()))
        dependents_label = st.selectbox("Dependents*", add_placeholder(binary_map.keys()))
    
        has_cash_card_label = st.selectbox("Has Cash Card*", add_placeholder(binary_map.keys()))
        multiple_lines_label = st.selectbox("Multiple Lines", add_placeholder(binary_map.keys()))
        digital_label = st.selectbox("Digital Access Level*", add_placeholder(internet_map.keys()))
        online_security_label = st.selectbox("Online Security*", add_placeholder(service_map.keys()))
        online_backup_label = st.selectbox("Online Backup", add_placeholder(service_map.keys()))
        device_protection_label = st.selectbox("Device Protection", add_placeholder(service_map.keys()))
        tech_support_label = st.selectbox("Tech Support*", add_placeholder(service_map.keys()))
        streaming_tv_label = st.selectbox("Streaming TV*", add_placeholder(streaming_map.keys()))
        streaming_movies_label = st.selectbox("Streaming Movies*", add_placeholder(streaming_map.keys()))
    
        subscription_label = st.selectbox("Subscription Type*", add_placeholder(subscription_map.keys()))
        paperless_label = st.selectbox("Paperless Billing*", add_placeholder(binary_map.keys()))
        payment_label = st.selectbox("Payment Method*", add_placeholder(payment_map.keys()))
    
        monthly_spend = st.number_input("Monthly Spend (ZAR)*", min_value=0.0, step=1.0, format="%.2f")
        total_transaction_value = st.number_input("Total Transaction Value (ZAR)*", min_value=0.0, step=1.0, format="%.2f")
        months_active = st.number_input("Months Active*", min_value=1, step=1)
        avg_spend_per_month = total_transaction_value / months_active if months_active > 0 else 0
    
        col1, col2 = st.columns([1, 1])
        with col1:
            submit = st.form_submit_button("🚀 Predict")
        with col2:
            reset = st.form_submit_button("🔄 Reset")


            if reset:
                st.experimental_rerun()
        
            if submit:
                # Validate all selections
                required_fields = [
                    gender_label, senior_label, partner_label, dependents_label,
                    has_cash_card_label, digital_label,
                    online_security_label, tech_support_label, streaming_tv_label, streaming_movies_label, subscription_label, paperless_label, payment_label, monthly_spend, total_transaction_value, months_active
                ]
            
                if "-- Select --" in required_fields:
                    st.error("⚠️ Please complete all fields before submitting.")
                else:
                    # Convert all to encoded values
                    input_data = pd.DataFrame([[
                        monthly_spend,
                        total_transaction_value,
                        months_active,
                        avg_spend_per_month,
                        gender_map[gender_label],
                        senior_map[senior_label],
                        binary_map[partner_label],
                        binary_map[dependents_label],
                        binary_map[has_cash_card_label],
                        binary_map[multiple_lines_label],
                        internet_map[digital_label],
                        service_map[online_security_label],
                        service_map[online_backup_label],
                        service_map[device_protection_label],
                        service_map[tech_support_label],
                        streaming_map[streaming_tv_label],
                        streaming_map[streaming_movies_label],
                        subscription_map[subscription_label],
                        binary_map[paperless_label],
                        payment_map[payment_label]
                    ]], columns=model.feature_names_in_)
            
                    churn_pred = model.predict(input_data)[0]
                    churn_prob = model.predict_proba(input_data)[0][1]
            
                    st.success(f"Prediction: {'🔴 Churn Risk' if churn_pred == 1 else '🟢 Not Likely to Churn'}")
                    st.info(f"Churn Probability: {churn_prob:.2%}")

# Optional: Show example format
with st.expander("View example data format"):
    st.write(pd.DataFrame(columns=feature_columns).head(1))
