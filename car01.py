# ======================================
# SMART PRICING SYSTEM FOR USED CARS - STREAMLIT READY
# ======================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Smart Car Pricing PRO", layout="wide")
st.title("🚗 Smart Pricing System for Used Cars - PRO")
st.markdown("### Upload your used car dataset and get AI-powered price predictions!")

sns.set(style="whitegrid")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    if 'Market_Price(INR)' not in df.columns:
        st.error("❌ Dataset must include 'Market_Price(INR)' column.")
        st.stop()

    # -------------------------------
    # Encode categorical columns
    # -------------------------------
    cat_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # -------------------------------
    # Model Training
    # -------------------------------
    X = df.drop(columns=['Market_Price(INR)'])
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
    with st.spinner("Training models..."):
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2 Score': r2_score(y_test, y_pred)
            }

    st.success("✅ Model training completed!")

    result_df = pd.DataFrame(results).T
    st.subheader("📈 Model Performance")
    st.dataframe(result_df)

    best_model_name = result_df['R2 Score'].idxmax()
    best_model = models[best_model_name]
    st.success(f"🏆 Best Model: {best_model_name}")

    # -------------------------------
    # Brand & Model Selection
    # -------------------------------
    st.subheader("💰 Predict Car Price")
    df_original = df.copy()
    for col in encoders:
        df_original[col] = encoders[col].inverse_transform(df[col])

    brand_list = sorted(df_original['Brand'].unique())
    selected_brand = st.selectbox("🚘 Select Brand", brand_list)
    model_list = sorted(df_original[df_original['Brand'] == selected_brand]['Model'].unique())
    selected_model = st.selectbox("🔧 Select Model", model_list)

    # -------------------------------
    # Auto-fill inputs
    # -------------------------------
    filtered = df_original[(df_original['Brand'] == selected_brand) & 
                           (df_original['Model'] == selected_model)]
    if not filtered.empty:
        base_data = filtered.iloc[0]
    else:
        base_data = pd.Series(dtype='object')

    inputs = {}
    for col in X.columns:
        if col in df_original.columns:
            if df_original[col].dtype == 'object':
                options = sorted(df_original[col].unique())
                default_val = base_data[col] if col in base_data else options[0]
                inputs[col] = st.selectbox(col, options, index=options.index(default_val) if default_val in options else 0)
            else:
                min_val, max_val = int(df_original[col].min()), int(df_original[col].max())
                default_val = int(base_data[col]) if col in base_data else int(df_original[col].median())
                inputs[col] = st.slider(col, min_value=min_val, max_value=max_val, value=default_val)

    # -------------------------------
    # Predict Price + Show Image
    # -------------------------------
    if st.button("🔍 Predict Price"):
        input_df = pd.DataFrame([inputs])
        for col in encoders:
            if col in input_df:
                input_df[col] = encoders[col].transform(input_df[col].astype(str))
        input_scaled = scaler.transform(input_df)
        predicted_price = best_model.predict(input_scaled)[0]

        st.subheader("📊 Price Estimation")
        st.metric("Minimum Price", f"₹{predicted_price*0.9:,.0f}")
        st.metric("Fair Market Price", f"₹{predicted_price:,.0f}")
        st.metric("Maximum Price", f"₹{predicted_price*1.1:,.0f}")

        # -------------------------------
        # Fetch car image from Unsplash
        # -------------------------------
        query = f"{selected_brand} {selected_model} car"
        access_key = "YOUR_UNSPLASH_ACCESS_KEY"  # <-- Sign up at unsplash.com and get key
        url = f"https://api.unsplash.com/photos/random?query={query}&client_id={access_key}&orientation=landscape"
        try:
            response = requests.get(url).json()
            image_url = response['urls']['regular']
            image = Image.open(BytesIO(requests.get(image_url).content))
            st.image(image, caption=f"{selected_brand} {selected_model}", use_column_width=True)
        except:
            st.info("🚗 Car image not available")
