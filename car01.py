# ======================================================
# SMART PRICING SYSTEM FOR USED CARS - FINAL STABLE VERSION
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

st.set_page_config(page_title="Smart Car Pricing PRO", layout="wide")
st.title("🚗 Smart Pricing System for Used Cars")
st.markdown("### Upload your dataset and get AI-powered price predictions & insights!")

sns.set(style="whitegrid")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV/XLSX File", type=["csv","xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            import openpyxl
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # Ensure price column exists
    if 'Market_Price(INR)' not in df.columns:
        st.error("❌ Dataset must include 'Market_Price(INR)' column.")
        st.stop()

    st.subheader("📊 Full Dataset")
    st.dataframe(df)

    # -------------------------------
    # Data Preprocessing
    # -------------------------------
    st.subheader("🧹 Data Cleaning & Encoding")
    df = df.dropna()
    cat_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # -------------------------------
    # Model Training
    # -------------------------------
    st.subheader("🤖 Model Training & Evaluation")
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
    result_df = pd.DataFrame(results).T
    st.subheader("📈 Model Performance Comparison")
    st.dataframe(result_df)

    best_model_name = result_df['R2 Score'].idxmax()
    best_model = models[best_model_name]
    st.success(f"🏆 Best Model Selected: **{best_model_name}**")

    # -------------------------------
    # Brand → Model → Prediction
    # -------------------------------
    st.subheader("💰 Predict Car Price")

    df_original = df.copy()
    for col in encoders:
        df_original[col] = encoders[col].inverse_transform(df[col])

    # Brand selection
    if 'Brand' in df_original.columns:
        brands = sorted(df_original['Brand'].unique())
        selected_brand = st.selectbox("🚘 Select Brand", brands)
        filtered_models = sorted(df_original[df_original['Brand'] == selected_brand]['Model'].unique())
        selected_model = st.selectbox("🔧 Select Model", filtered_models)
    else:
        st.warning("⚠ 'Brand' column not found, showing manual input only.")
        selected_brand, selected_model = None, None

    # Filter data
    filtered_rows = df_original[(df_original['Brand'] == selected_brand) &
                                (df_original['Model'] == selected_model)] if selected_brand and selected_model else df_original

    # Show car images (if dataset has Image_URL column)
    if 'Image_URL' in df_original.columns and not filtered_rows.empty:
        st.markdown("### 🖼️ Car Gallery")
        cols = st.columns(3)
        for i, (_, row) in enumerate(filtered_rows.head(3).iterrows()):
            cols[i % 3].image(row['Image_URL'], use_container_width=True,
                              caption=f"{row['Brand']} {row['Model']}")

    st.markdown("### 🧩 Car Details (Auto-filled + Editable)")
    inputs = {}
    for col in feature_columns:
        if col in df_original.columns:
            if df_original[col].dtype == 'object':
                options = sorted(df_original[col].unique())
                default = filtered_rows[col].iloc[0] if not filtered_rows.empty else options[0]
                inputs[col] = st.selectbox(f"{col}", options, index=options.index(default))
            else:
                min_val, max_val = int(df_original[col].min()), int(df_original[col].max())
                default_val = int(filtered_rows[col].iloc[0]) if not filtered_rows.empty else int(df_original[col].mean())
                inputs[col] = st.slider(f"{col}", min_val, max_val, default_val)
        else:
            inputs[col] = 0

    if st.button("🔍 Predict Price"):
        input_df = pd.DataFrame([inputs])
        for col in encoders:
            if col in input_df:
                input_df[col] = encoders[col].transform(input_df[col].astype(str))
        input_scaled = scaler.transform(input_df)
        predicted_price = best_model.predict(input_scaled)[0]

        st.subheader("📊 Price Estimation")
        st.metric("Minimum Negotiation Price", f"₹{predicted_price*0.9:,.0f}")
        st.metric("Fair Market Price", f"₹{predicted_price:,.0f}")
        st.metric("Maximum Negotiation Price", f"₹{predicted_price*1.1:,.0f}")
        st.balloons()

    # -------------------------------
    # Market Insights
    # -------------------------------
    st.subheader("📉 Market Insights")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df_original['Market_Price(INR)'], kde=True, ax=ax)
        ax.set_title("Distribution of Market Prices")
        st.pyplot(fig)

    with col2:
        if 'Fuel_Type' in df_original.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x='Fuel_Type', y='Market_Price(INR)', data=df_original, ax=ax)
            ax.set_title("Fuel Type vs Price")
            st.pyplot(fig)

else:
    st.info("📥 Please upload your dataset to start.")
