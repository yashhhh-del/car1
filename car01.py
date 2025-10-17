# ======================================================
# SMART PRICING SYSTEM FOR USED CARS - STREAMLIT PRO
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
import pickle

st.set_page_config(page_title="Smart Car Pricing PRO", layout="wide")
st.title("🚗 Smart Pricing System for Used Cars - PRO")
st.markdown("### Upload your dataset and get AI-powered price predictions & market insights!")

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
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # -------------------------------
    # Model Training
    # -------------------------------
    st.subheader("🤖 Model Training & Evaluation")
    X = df.drop(columns=['Market_Price(INR)'], errors='ignore')
    y = df['Market_Price(INR)']

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

    # Display model results
    st.subheader("📈 Model Performance Comparison")
    result_df = pd.DataFrame(results).T
    st.dataframe(result_df)

    best_model_name = result_df['R2 Score'].idxmax()
    best_model = models[best_model_name]
    st.success(f"🏆 Best Model Selected: **{best_model_name}**")

    # -------------------------------
    # Feature Importance (for tree-based models)
    # -------------------------------
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        st.subheader("🌟 Feature Importance")
        importance = best_model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
        st.bar_chart(fi_df.set_index('Feature'))

    # -------------------------------
    # Price Prediction Form with dropdowns & sliders
    # -------------------------------
    st.subheader("💰 Predict Car Price")

    with st.form("price_form"):
        inputs = {}
        for col in feature_columns:
            if col in encoders:
                # dropdown for categorical
                inv_map = {v: k for k, v in encoders[col].classes_.items()} if hasattr(encoders[col], 'classes_') else {}
                inputs[col] = st.selectbox(f"{col}", options=list(range(len(encoders[col].classes_))))
            else:
                # slider for numerical
                min_val = int(df[col].min())
                max_val = int(df[col].max())
                default_val = int(df[col].median())
                inputs[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=default_val)

        submit_btn = st.form_submit_button("🔍 Predict Price")

    if submit_btn:
        input_df = pd.DataFrame([list(inputs.values())], columns=feature_columns)
        input_scaled = scaler.transform(input_df)
        predicted_price = best_model.predict(input_scaled)[0]

        min_price = predicted_price * 0.9
        mid_price = predicted_price
        max_price = predicted_price * 1.1

        st.subheader("📊 Price Estimation")
        st.metric("Minimum Negotiation Price", f"₹{min_price:,.0f}")
        st.metric("Fair Market Price", f"₹{mid_price:,.0f}")
        st.metric("Maximum Negotiation Price", f"₹{max_price:,.0f}")

        # Suggestion
        st.subheader("💡 Deal Suggestion")
        st.info("💰 Good deal if below min, overpriced if above max, fair in between.")

        # Download prediction
        download_df = input_df.copy()
        download_df['Predicted_Price'] = predicted_price
        download_df['Min_Price'] = min_price
        download_df['Mid_Price'] = mid_price
        download_df['Max_Price'] = max_price
        st.download_button("⬇️ Download Prediction CSV", download_df.to_csv(index=False), file_name="prediction.csv")

    # -------------------------------
    # Market Insights & Visualization
    # -------------------------------
    st.subheader("📉 Market Insights & Visualization")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df['Market_Price(INR)'], kde=True, ax=ax)
        ax.set_title("Distribution of Market Prices")
        st.pyplot(fig)

    with col2:
        cat_for_box = None
        for c in ['Fuel_Type', 'Transmission', 'Owner']:
            if c in df.columns:
                cat_for_box = c
                break
        if cat_for_box:
            fig, ax = plt.subplots()
            sns.boxplot(x=cat_for_box, y='Market_Price(INR)', data=df, ax=ax)
            ax.set_title(f"{cat_for_box} vs Market Price")
            st.pyplot(fig)

else:
    st.info("📥 Please upload your dataset to start.")
