# ======================================================
# SMART PRICING SYSTEM FOR USED CARS - ADVANCED UI EDITION
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Smart Car Pricing Pro", layout="wide")
st.title("🚗 Smart Pricing System for Used Cars - Advanced Edition")
st.markdown("### Upload your dataset and explore AI-powered car pricing insights 🚀")

sns.set(style="whitegrid")

# ---------------------------------
# File Upload
# ---------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(df)

    if 'Market_Price(INR)' not in df.columns:
        st.error("❌ Dataset must include 'Market_Price(INR)' column.")
        st.stop()

    # -------------------------------
    # Data Preprocessing
    # -------------------------------
    st.subheader("🧹 Data Cleaning & Encoding")
    df = df.dropna()

    cat_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=['Market_Price(INR)'])
    y = df['Market_Price(INR)']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_columns = X.columns.tolist()

    # -------------------------------
    # Model Training
    # -------------------------------
    st.subheader("🤖 Model Training & Evaluation")
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

    result_df = pd.DataFrame(results).T
    st.dataframe(result_df)

    best_model_name = result_df['R2 Score'].idxmax()
    best_model = models[best_model_name]
    st.success(f"🏆 Best Model Selected: **{best_model_name}**")

    # -------------------------------
    # Brand → Model → Auto-fill Flow
    # -------------------------------
    st.subheader("🚘 Car Selection & Price Prediction")

    df_original = df.copy()
    for col in encoders:
        df_original[col] = encoders[col].inverse_transform(df[col])

    brand_list = sorted(df_original['Brand'].unique())
    selected_brand = st.selectbox("Select Car Brand", brand_list)

    model_list = sorted(df_original[df_original['Brand'] == selected_brand]['Model'].unique())
    selected_model = st.selectbox("Select Car Model", model_list)

    car_data = df_original[(df_original['Brand'] == selected_brand) & (df_original['Model'] == selected_model)]
    if not car_data.empty:
        base_data = car_data.iloc[0]

        st.markdown("### 🧩 Auto-Filled Car Details (Editable)")
        cols = st.columns(3)
        user_inputs = {}

        # Create editable inputs (auto-filled)
        for i, col in enumerate(['Car_Type', 'Fuel_Type', 'Transmission', 'Year', 'Mileage(km)',
                                 'Engine_cc', 'Power_HP', 'Seats', 'Condition', 'Owner',
                                 'Insurance_Status', 'Registration_City', 'Service_History',
                                 'Accident_History', 'Car_Availability']):
            with cols[i % 3]:
                if col in df_original.columns:
                    if df_original[col].dtype == 'object':
                        options = sorted(df_original[col].unique())
                        user_inputs[col] = st.selectbox(col, options,
                                                        index=options.index(base_data[col]) if base_data[col] in options else 0)
                    else:
                        user_inputs[col] = st.number_input(col, value=float(base_data[col]), step=1.0)

        # -------------------------------
        # Prediction
        # -------------------------------
        if st.button("🔍 Predict Price"):
            input_data = base_data.copy()
            for k, v in user_inputs.items():
                input_data[k] = v

            # Encode
            input_encoded = input_data.copy()
            for col in encoders:
                if col in input_encoded:
                    input_encoded[col] = encoders[col].transform([str(input_encoded[col])])[0]

            input_encoded = pd.DataFrame([input_encoded]).reindex(columns=feature_columns, fill_value=0)
            input_scaled = scaler.transform(input_encoded)

            predicted_price = best_model.predict(input_scaled)[0]

            st.subheader("📊 Price Estimation")
            st.metric("Minimum Negotiation Price", f"₹{predicted_price*0.9:,.0f}")
            st.metric("Fair Market Price", f"₹{predicted_price:,.0f}")
            st.metric("Maximum Negotiation Price", f"₹{predicted_price*1.1:,.0f}")

            st.balloons()

    # -------------------------------
    # Visualization
    # -------------------------------
    st.subheader("📉 Market Insights")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df_original['Market_Price(INR)'], kde=True, ax=ax)
        ax.set_title("Market Price Distribution")
        st.pyplot(fig)

    with col2:
        if 'Fuel_Type' in df_original.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x='Fuel_Type', y='Market_Price(INR)', data=df_original, ax=ax)
            ax.set_title("Fuel Type vs Market Price")
            st.pyplot(fig)

else:
    st.info("📥 Please upload your dataset to start.")
