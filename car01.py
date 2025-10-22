# ======================================================
# SMART PRICING SYSTEM - AUTO IMAGE (No API Needed)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Smart Car Pricing System", layout="wide")
st.title("🚗 Smart Pricing System for Used Cars")
st.markdown("### AI-Powered Price Estimator with Auto Image Fetch (No API Needed)")

sns.set(style="whitegrid")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV/XLSX File", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        import openpyxl
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    st.success("✅ File uploaded successfully!")

    # Ensure Market_Price column
    if 'Market_Price(INR)' not in df.columns:
        st.warning("⚠ 'Market_Price(INR)' column missing! Added random demo prices.")
        df['Market_Price(INR)'] = np.random.randint(300000, 2000000, size=len(df))

    st.subheader("📊 Preview Dataset")
    st.dataframe(df.head())

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Train model
    X = df.drop(columns=['Market_Price(INR)'])
    y = df['Market_Price(INR)']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.success(f"📈 Model trained successfully (R² = {r2_score(y_test, y_pred):.2f})")

    # Decode data for display
    df_original = pd.DataFrame()
    for col in encoders:
        df_original[col] = encoders[col].inverse_transform(df[col])
    df_original['Market_Price(INR)'] = y

    # -------------------------------
    # Brand & Model Selection
    # -------------------------------
    st.subheader("🎯 Select Your Car")

    brands = sorted(df_original['Brand'].unique())
    selected_brand = st.selectbox("🚘 Select Brand", brands)

    models = sorted(df_original[df_original['Brand'] == selected_brand]['Model'].unique())
    selected_model = st.selectbox("🔧 Select Model", models)

    # Auto Image (Unsplash without API)
    image_url = f"https://source.unsplash.com/800x400/?{selected_brand}%20{selected_model}%20car"
    st.image(image_url, caption=f"{selected_brand} {selected_model}", use_container_width=True)

    # -------------------------------
    # Dataset-based Price Stats
    # -------------------------------
    car_data = df_original[(df_original['Brand'] == selected_brand) &
                           (df_original['Model'] == selected_model)]

    if not car_data.empty:
        min_price = car_data['Market_Price(INR)'].min()
        avg_price = car_data['Market_Price(INR)'].mean()
        max_price = car_data['Market_Price(INR)'].max()

        st.markdown("### 💰 Dataset Price Range")
        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum", f"₹{min_price:,.0f}")
        col2.metric("Average", f"₹{avg_price:,.0f}")
        col3.metric("Maximum", f"₹{max_price:,.0f}")
    else:
        st.warning("⚠ No matching car data found in dataset!")

    # -------------------------------
    # Predict Car Price (Manual Input)
    # -------------------------------
    st.subheader("🧮 Predict Your Car’s Price")

    inputs = {}
    for col in X.columns:
        if col in encoders:
            options = sorted(df_original[col].unique())
            inputs[col] = st.selectbox(col, options)
        else:
            val = float(st.number_input(f"{col}", value=float(df_original[col].mean())))
            inputs[col] = val

    if st.button("🔍 Predict Price"):
        input_df = pd.DataFrame([inputs])
        for col in encoders:
            if col in input_df:
                input_df[col] = encoders[col].transform(input_df[col].astype(str))
        input_scaled = scaler.transform(input_df)
        predicted_price = model.predict(input_scaled)[0]

        st.image(image_url, caption="Predicted Car Image", use_container_width=True)
        st.subheader("🏷 Predicted Price Range")
        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum", f"₹{predicted_price * 0.9:,.0f}")
        col2.metric("Estimated", f"₹{predicted_price:,.0f}")
        col3.metric("Maximum", f"₹{predicted_price * 1.1:,.0f}")
        st.balloons()
else:
    st.info("📥 Upload a dataset to begin.")
