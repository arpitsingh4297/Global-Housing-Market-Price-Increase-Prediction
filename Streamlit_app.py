import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Global Housing Market Price Prediction", layout="wide")

# Title and description
st.title("🏠 Global Housing Market Price Increase Prediction")
st.markdown("""
This app predicts whether house prices will increase based on economic and market features.
Enter the required features below, select a country, and get a prediction using a trained Logistic Regression model.
""")

# Load model and scaler
try:
    model = pickle.load(open("logistic_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'logistic_model.pkl' and 'scaler.pkl' are in the directory.")
    st.stop()

# Define feature names and countries
numerical_features = [
    "Rent Index", "Affordability Ratio", "Mortgage Rate (%)",
    "Inflation Rate (%)", "GDP Growth (%)", "Population Growth (%)",
    "Urbanization Rate (%)", "Construction Index", "House Price Index_lag"
]
countries = [
    "Australia", "Brazil", "Canada", "China", "France", "Germany", "India",
    "Italy", "Japan", "Mexico", "Netherlands", "Russia", "South Africa",
    "South Korea", "Spain", "Sweden", "Switzerland", "UAE", "UK", "USA"
]

# Input form
st.header("📝 Input Features")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    # Numerical inputs
    with col1:
        rent_index = st.number_input(
            "Rent Index", min_value=0.0, max_value=200.0, value=21.0, step=0.1
        )
        affordability_ratio = st.number_input(
            "Affordability Ratio", min_value=0.0, max_value=20.0, value=13.0, step=0.1
        )
        mortgage_rate = st.number_input(
            "Mortgage Rate (%)", min_value=0.0, max_value=15.0, value=7.0, step=0.1
        )
    
    with col2:
        inflation_rate = st.number_input(
            "Inflation Rate (%)", min_value=-5.0, max_value=20.0, value=10.0, step=0.1
        )
        gdp_growth = st.number_input(
            "GDP Growth (%)", min_value=-10.0, max_value=15.0, value=3.0, step=0.1
        )
        population_growth = st.number_input(
            "Population Growth (%)", min_value=-5.0, max_value=10.0, value=5.0, step=0.1
        )
    
    with col3:
        urbanization_rate = st.number_input(
            "Urbanization Rate (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1
        )
        construction_index = st.number_input(
            "Construction Index", min_value=0.0, max_value=200.0, value=100.0, step=0.1
        )
        house_price_lag = st.number_input(
            "House Price Index (Previous Year)", min_value=0.0, max_value=300.0, value=75.0, step=0.1
        )
    
    # Country selection
    country = st.selectbox("Country", countries)
    
    # Submit button
    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    st.header("🔍 Prediction Results")
    
    # Prepare input data
    input_data = {
        "Rent Index": rent_index,
        "Affordability Ratio": affordability_ratio,
        "Mortgage Rate (%)": mortgage_rate,
        "Inflation Rate (%)": inflation_rate,
        "GDP Growth (%)": gdp_growth,
        "Population Growth (%)": population_growth,
        "Urbanization Rate (%)": urbanization_rate,
        "Construction Index": construction_index,
        "House Price Index_lag": house_price_lag
    }
    
    # Create one-hot encoded country columns
    for c in countries[1:]:  # Exclude first country (Australia) for drop_first=True
        input_data[f"Country_{c}"] = 1 if c == country else 0
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Scale numerical features
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    
    # Ensure all expected columns are present
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training
    input_df = input_df[expected_columns]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # Display results
    st.subheader("Prediction")
    st.write(f"**Price Increase**: {'Yes' if prediction == 1 else 'No'}")
    st.write(f"**Probability of Price Increase**: {prediction_proba[1]:.2%}")
    st.write(f"**Probability of No Increase**: {prediction_proba[0]:.2%}")
    
    # Plot probability
    st.subheader("Prediction Probability")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=["No Increase", "Price Increase"], y=prediction_proba)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    
    # Feature importance (Logistic Regression coefficients)
    st.subheader("Feature Importance")
    coef_df = pd.DataFrame({
        "Feature": expected_columns,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", key=abs, ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Coefficient", y="Feature", data=coef_df)
    ax.set_title("Top Features by Coefficient Magnitude")
    st.pyplot(fig)
    
    # Input feature distribution
    st.subheader("Input Features Overview")
    input_melted = pd.DataFrame({
        "Feature": numerical_features,
        "Value": [input_data[f] for f in numerical_features]
    })
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Value", y="Feature", data=input_melted)
    ax.set_title("Input Feature Values")
    st.pyplot(fig)

# Instructions
st.header("ℹ️ How to Use")
st.markdown("""
1. Enter the numerical values for the housing market features (e.g., Rent Index, House Price Index from the previous year).
2. Select a country from the dropdown.
3. Click **Predict** to see if house prices are likely to increase.
4. View the prediction, probability, and feature importance plots.
""")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Model: Logistic Regression | Data: Global Housing Market (2015–2024)")