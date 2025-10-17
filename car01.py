# ======================================================
# SMART PRICING SYSTEM FOR USED CARS - STREAMLIT READY
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit config
st.set_page_config(page_title="Smart Pricing System for Used Cars", layout="wide")
st.title("🚗 Smart Pricing System for Used Cars")
st.markdown("### Upload your used car dataset and get AI-powered price predictions!")

sns.set(style="whitegrid")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Show full dataset
    st.subheader("📊 Full Dataset")
    st.dataframe(df)

    # -------------------------------
    # Data Preprocessing
    # -------------------------------
    st.subheader("🧹 Data Cleaning & Preprocessing")
    df = df.dropna()

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # -------------------------------
    # Model Training
    # -------------------------------
    st.subheader("🤖 Model Training & Evaluation")
    X = df.drop(columns=['Market_Price(INR)'], errors='ignore')
    y = df['Market_Price(INR)']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}

    with st.spinner("Training models, please wait..."):
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2 Score': r2_score(y_test, y_pred)
            }

    st.success("✅ Model training completed!")

    # -------------------------------
    # Display Model Results
    # -------------------------------
    st.subheader("📈 Model Performance Comparison")
    result_df = pd.DataFrame(results).T
    st.dataframe(result_df)

    # Best Model Selection
    best_model_name = result_df['R2 Score'].idxmax()
    best_model = models[best_model_name]
    st.success(f"🏆 Best Model Selected: **{best_model_name}**")

    # -------------------------------
    # Price Prediction Form
    # -------------------------------
    st.subheader("💰 Predict Car Price")
    input_columns = ['Brand', 'Model', 'Year', 'Age', 'Mileage(km)', 'Fuel_Type',
                     'Transmission', 'Owner', 'Location']

    with st.form("price_form"):
        brand = st.number_input("Brand (encoded value)", min_value=0, value=0)
        model_name = st.number_input("Model (encoded value)", min_value=0, value=0)
        year = st.number_input("Year", min_value=1990, max_value=2025, value=2018)
        age = st.number_input("Car Age (years)", min_value=0, value=5)
        mileage = st.number_input("Mileage (km)", min_value=0, value=50000)
        fuel = st.number_input("Fuel Type (encoded)", min_value=0, value=0)
        transmission = st.number_input("Transmission (encoded)", min_value=0, value=0)
        owner = st.number_input("Owner (encoded)", min_value=0, value=0)
        location = st.number_input("Location (encoded)", min_value=0, value=0)

        submit_btn = st.form_submit_button("🔍 Predict Price")

    if submit_btn:
        input_data = pd.DataFrame([[brand, model_name, year, age, mileage, fuel, transmission, owner, location]],
                                  columns=input_columns)
        input_scaled = scaler.transform(input_data)
        predicted_price = best_model.predict(input_scaled)[0]

        # Min, Mid, Max price
        min_price = predicted_price * 0.9
        mid_price = predicted_price
        max_price = predicted_price * 1.1

        st.subheader("📊 Price Estimation")
        st.metric("Minimum Negotiation Price", f"₹{min_price:,.0f}")
        st.metric("Fair Market Price", f"₹{mid_price:,.0f}")
        st.metric("Maximum Negotiation Price", f"₹{max_price:,.0f}")

        st.balloons()

    # -------------------------------
    # Visualization
    # -------------------------------
    st.subheader("📉 Price Insights & Visualization")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df['Market_Price(INR)'], kde=True, ax=ax)
        ax.set_title("Distribution of Market Prices")
        st.pyplot(fig)

    with col2:
        if 'Fuel_Type' in df.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x='Fuel_Type', y='Market_Price(INR)', data=df, ax=ax)
            ax.set_title("Fuel Type vs Market Price")
            st.pyplot(fig)

else:
    st.info("📥 Please upload your dataset to start.")
