# Fintech Customer Churn Prediction

This project predicts customer churn for a fintech app or platform using historical customer data (adapted from Telco Churn Dataset). Built with Python, scikit-learn, and deployed via Streamlit.

## Structure

- `data/`: Raw and cleaned datasets
- `notebooks/`: EDA, feature engineering, and modelling notebooks
- `app/`: Streamlit app and saved model
- `requirements.txt`: Package list for easy setup

## Notebooks

- `01_explore_data.ipynb`: Load, clean, and visualise churn trends
- `02_feature_engineering.ipynb`: Encode, scale, and prep features
- `03_modeling.ipynb`: Train and evaluate ML models

## Streamlit App

Launch with:

```bash
cd app
streamlit run app.py
