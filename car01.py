# ======================================================
# SMART PRICING SYSTEM FOR USED CARS - AUTO PRICE DETECT
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

    st.subheader("📊 Full Dataset")
    st.dataframe(df)

    # -------------------------------
    # Detect Price Column Automatically
    # -------------------------------
    target_col = None
    for col in df.columns:
        if 'price' in col.lower():
            target_col = col
            break

    if target_col is None:
        st.error("❌ Dataset must include a price column (like Market_Price(INR), Selling_Price, or Price).")
        st.stop()
    else:
        st.success(f"✅ Using '{target_col}' as target column for model training.")

    # -------------------------------
    # Data Preprocessing
    # -------------------------------
    st.subheader("🧹 Data Cleaning & Preprocessing")
    df = df.dropna()

    cat_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # -------------------------------
    # Model Training
    # -------------------------------
    st.subheader("🤖 Model Training & Evaluation")

    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    with st.spinner("⏳ Training models, please wait..."):
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2 Score': r2_score(y_test, y_pred)
            }

    st.success("✅ Model training completed successfully!")

    # Display model results
    st.subheader("📈 Model Performance Comparison")
    result_df = pd.DataFrame(results).T
    st.dataframe(result_df.style.highlight_max(axis=0, color='lightgreen'))

    best_model_name = result_df['R2 Score'].idxmax()
    best_model = models[best_model_name]
    st.success(f"🏆 Best Model Selected: **{best_model_name}**")

    # -------------------------------
    # Price Prediction Form
    # -------------------------------
    st.subheader("💰 Predict Car Price")
    with st.form("price_form"):
        inputs = {}
        for col in feature_columns:
            inputs[col] = st.number_input(f"{col}", min_value=0.0, value=0.0)

        submit_btn = st.form_submit_button("🔍 Predict Price")

    if submit_btn:
        input_df = pd.DataFrame([list(inputs.values())], columns=feature_columns)
        input_scaled = scaler.transform(input_df)
        predicted_price = best_model.predict(input_scaled)[0]

        min_price = predicted_price * 0.9
        mid_price = predicted_price
        max_price = predicted_price * 1.1

        st.subheader("📊 Price Estimation")
        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum Negotiation Price", f"₹{min_price:,.0f}")
        col2.metric("Fair Market Price", f"₹{mid_price:,.0f}")
        col3.metric("Maximum Negotiation Price", f"₹{max_price:,.0f}")
        st.balloons()

    # -------------------------------
    # Visualization
    # -------------------------------
    st.subheader("📉 Market Insights & Visualization")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df[target_col], kde=True, ax=ax)
        ax.set_title("Distribution of Market Prices")
        st.pyplot(fig)

    with col2:
        if 'Fuel_Type' in df.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x='Fuel_Type', y=target_col, data=df, ax=ax)
            ax.set_title("Fuel Type vs Market Price")
            st.pyplot(fig)

else:
    st.info("📥 Please upload your dataset to start.")
