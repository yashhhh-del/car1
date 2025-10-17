# ======================================================
# SMART PRICING SYSTEM FOR USED CARS - STREAMLIT PRO
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Smart Car Pricing PRO", layout="wide")
st.title("🚗 Smart Pricing System for Used Cars - PRO")
st.markdown("### Upload your dataset to get AI-powered price predictions & market insights!")

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
    # Check required columns
    # -------------------------------
    required_cols = ['Brand','Model','Fuel_Type','Transmission','Owner','Age','Mileage(km)','Market_Price(INR)']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"⚠ Uploaded CSV is missing these columns: {missing_cols}")
        st.stop()

    # -------------------------------
    # Preprocessing
    # -------------------------------
    st.subheader("🧹 Data Cleaning & Preprocessing")
    df = df.dropna()
    encoders = {}
    cat_cols = ['Brand','Model','Fuel_Type','Transmission','Owner']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    scaler = StandardScaler()
    num_cols = ['Age','Mileage(km)']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # -------------------------------
    # Model Training
    # -------------------------------
    st.subheader("🤖 Model Training & Evaluation")
    X = df.drop(columns=['Market_Price(INR)'])
    y = df['Market_Price(INR)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2 Score': r2_score(y_test, y_pred)
        }

    result_df = pd.DataFrame(results).T
    st.subheader("📈 Model Performance")
    st.dataframe(result_df)

    best_model_name = result_df['R2 Score'].idxmax()
    best_model = models[best_model_name]
    st.success(f"🏆 Best Model: {best_model_name}")

    # -------------------------------
    # Price Prediction Form
    # -------------------------------
    st.subheader("💰 Predict Car Price")
    
    # Dynamic Brand → Model selection
    inv_brand = {i: cls for i, cls in enumerate(encoders['Brand'].classes_)}
    selected_brand = st.selectbox("Brand", options=list(inv_brand.keys()), format_func=lambda x: inv_brand[x])

    # Filter models for selected brand instantly
    df_temp = df.copy()
    df_temp['Brand_orig'] = encoders['Brand'].inverse_transform(df_temp['Brand'])
    df_temp['Model_orig'] = encoders['Model'].inverse_transform(df_temp['Model'])
    models_for_brand = df_temp[df_temp['Brand_orig'] == inv_brand[selected_brand]]['Model_orig'].unique()
    selected_model = st.selectbox("Model", options=models_for_brand)

    # Encode selected brand & model
    brand_enc = encoders['Brand'].transform([inv_brand[selected_brand]])[0]
    model_enc = encoders['Model'].transform([selected_model])[0]

    # Other features input
    input_data = {'Brand': brand_enc, 'Model': model_enc}
    for col in ['Fuel_Type','Transmission','Owner','Age','Mileage(km)']:
        if col in encoders:
            options = list(encoders[col].classes_)
            selected = st.selectbox(col, options=options)
            input_data[col] = encoders[col].transform([selected])[0]
        else:
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            default_val = int(df[col].median())
            input_data[col] = st.slider(col, min_value=min_val, max_value=max_val, value=default_val)

    if st.button("🔍 Predict Price"):
        input_df = pd.DataFrame([list(input_data.values())], columns=X.columns)
        predicted_price = best_model.predict(input_df)[0]
        min_price = predicted_price*0.9
        mid_price = predicted_price
        max_price = predicted_price*1.1

        st.subheader("📊 Price Estimation")
        st.metric("Minimum Negotiation Price", f"₹{min_price:,.0f}")
        st.metric("Fair Market Price", f"₹{mid_price:,.0f}")
        st.metric("Maximum Negotiation Price", f"₹{max_price:,.0f}")

    # -------------------------------
    # Market Insights & Visualization
    # -------------------------------
    st.subheader("📉 Market Insights")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df['Market_Price(INR)'], kde=True, ax=ax)
        ax.set_title("Distribution of Market Prices")
        st.pyplot(fig)

    with col2:
        cat_col = None
        for c in ['Fuel_Type','Transmission','Owner']:
            if c in df.columns:
                cat_col = c
                break
        if cat_col:
            fig, ax = plt.subplots()
            sns.boxplot(x=cat_col, y='Market_Price(INR)', data=df, ax=ax)
            ax.set_title(f"{cat_col} vs Market Price")
            st.pyplot(fig)

else:
    st.info("📥 Please upload your CSV to start!")
