# ======================================================
# SMART PRICING SYSTEM FOR USED CARS - ADVANCED VERSION
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Smart Pricing System for Used Cars", layout="wide")
st.title("🚗 Smart Pricing System for Used Cars")
st.markdown("### Upload your used car dataset and get **AI-powered price predictions & insights!**")

sns.set(style="whitegrid")

MODEL_FILE = "best_car_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODERS_FILE = "encoders.pkl"
FEATURES_FILE = "features.pkl"

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")
        st.stop()

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Check if saved model exists
    # -------------------------------
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        st.info("📦 Loading saved model and scaler (skipping training)...")
        best_model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        encoders = joblib.load(ENCODERS_FILE)
        feature_columns = joblib.load(FEATURES_FILE)
        st.success("✅ Model loaded successfully!")

    else:
        # -------------------------------
        # Data Preprocessing
        # -------------------------------
        st.subheader("🧹 Data Cleaning & Preprocessing")

        df = df.dropna()
        st.write(f"✅ Removed missing values. Final rows: {len(df)}")

        cat_cols = df.select_dtypes(include=['object']).columns
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        st.write("✅ Encoded categorical columns:", list(cat_cols))

        # -------------------------------
        # Model Training
        # -------------------------------
        st.subheader("🤖 Model Training & Evaluation")

        if 'Market_Price(INR)' not in df.columns:
            st.error("❌ Dataset must include 'Market_Price(INR)' column.")
            st.stop()

        X = df.drop(columns=['Market_Price(INR)'], errors='ignore')
        y = df['Market_Price(INR)']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        results = {}
        with st.spinner("Training models..."):
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = {
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'R2 Score': r2_score(y_test, y_pred)
                }

        result_df = pd.DataFrame(results).T
        st.success("✅ Model training completed!")

        st.subheader("📈 Model Performance Comparison")
        st.dataframe(result_df.style.format("{:.4f}"))

        best_model_name = result_df['R2 Score'].idxmax()
        best_model = models[best_model_name]
        st.success(f"🏆 Best Model Selected: **{best_model_name}**")

        # Save model files
        joblib.dump(best_model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        joblib.dump(encoders, ENCODERS_FILE)
        joblib.dump(feature_columns, FEATURES_FILE)
        st.info("💾 Model & Preprocessing Objects Saved for Future Use!")

    # -------------------------------
    # Price Prediction Form
    # -------------------------------
    st.subheader("💰 Predict Car Price")

    with st.form("price_form"):
        st.write("### Enter Car Details for Prediction")

        inputs = {}
        for col in feature_columns:
            if col in encoders:
                options = list(encoders[col].classes_)
                selected = st.selectbox(f"{col}", options)
                encoded_value = encoders[col].transform([selected])[0]
                inputs[col] = encoded_value
            else:
                inputs[col] = st.number_input(f"{col}", min_value=0.0, value=0.0)

        submit_btn = st.form_submit_button("🔍 Predict Price")

    if submit_btn:
        input_df = pd.DataFrame([inputs])
        input_scaled = scaler.transform(input_df)
        predicted_price = best_model.predict(input_scaled)[0]

        st.subheader("📊 Price Estimation")
        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum Negotiation Price", f"₹{predicted_price * 0.9:,.0f}")
        col2.metric("Fair Market Price", f"₹{predicted_price:,.0f}")
        col3.metric("Maximum Negotiation Price", f"₹{predicted_price * 1.1:,.0f}")
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
