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
import os

st.set_page_config(page_title="Smart Car Pricing PRO", layout="wide")
st.title("🚗 Smart Pricing System for Used Cars - PRO")
st.markdown("### Upload your dataset or use default dataset to get AI-powered price predictions & market insights!")

sns.set(style="whitegrid")

# -------------------------------
# Load Default Dataset
# -------------------------------
default_file_path = './All_Car_Types_Full_Dataset.xlsx'

if os.path.exists(default_file_path):
    default_df = pd.read_excel(default_file_path)
else:
    default_df = None

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV or Excel File", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success("✅ File uploaded successfully!")
elif default_df is not None:
    df = default_df.copy()
    st.info("Using default dataset (All car types included).")
else:
    st.warning("No dataset available. Please upload a file.")
    st.stop()

# -------------------------------
# Show Full Dataset
# -------------------------------
st.subheader("📊 Full Dataset")
st.dataframe(df)

# -------------------------------
# Preprocessing
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

# Scale numerical columns
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
num_cols = [c for c in num_cols if c not in ['Car_ID', 'Price_INR']]  # Exclude ID and target
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# -------------------------------
# Model Training
# -------------------------------
st.subheader("🤖 Model Training & Evaluation")
X = df.drop(columns=['Price_INR'], errors='ignore')
y = df['Price_INR']

feature_columns = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
st.subheader("📈 Model Performance Comparison")
result_df = pd.DataFrame(results).T
st.dataframe(result_df)

best_model_name = result_df['R2 Score'].idxmax()
best_model = models[best_model_name]
st.success(f"🏆 Best Model Selected: **{best_model_name}**")

# -------------------------------
# Feature Importance (if tree-based)
# -------------------------------
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    st.subheader("🌟 Feature Importance")
    importance = best_model.feature_importances_
    fi_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    st.bar_chart(fi_df.set_index('Feature'))

# -------------------------------
# Price Prediction Form
# -------------------------------
st.subheader("💰 Predict Car Price")

brand_col = 'Brand'
model_col = 'Model'

# Dynamic Brand -> Model
inv_brand = {i: cls for i, cls in enumerate(encoders[brand_col].classes_)}
selected_brand = st.selectbox(f"{brand_col}", options=list(inv_brand.keys()), format_func=lambda x: inv_brand[x])

df_original = df.copy()
df_original[brand_col] = df[brand_col].map(lambda x: encoders[brand_col].inverse_transform([x])[0])
df_original[model_col] = df[model_col].map(lambda x: encoders[model_col].inverse_transform([x])[0])

models_for_brand = df_original[df_original[brand_col] == inv_brand[selected_brand]][model_col].unique()
models_for_brand_encoded = [encoders[model_col].transform([m])[0] for m in models_for_brand]
selected_model = st.selectbox(f"{model_col}", options=models_for_brand_encoded,
                              format_func=lambda x: encoders[model_col].inverse_transform([x])[0])

# Other features dynamically
inputs = {brand_col: selected_brand, model_col: selected_model}
for col in feature_columns:
    if col not in [brand_col, model_col]:
        if col in encoders:
            inv_map = {i: cls for i, cls in enumerate(encoders[col].classes_)}
            inputs[col] = st.selectbox(f"{col}", options=list(inv_map.keys()), format_func=lambda x: inv_map[x])
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            default_val = float(df[col].median())
            inputs[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=default_val)

if st.button("🔍 Predict Price"):
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

    st.subheader("💡 Deal Suggestion")
    st.info("💰 Good deal if below min, overpriced if above max, fair in between.")

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
    sns.histplot(df['Price_INR'], kde=True, ax=ax)
    ax.set_title("Distribution of Market Prices")
    st.pyplot(fig)

with col2:
    cat_for_box = None
    for c in ['Fuel_Type', 'Transmission', 'Owner', 'Insurance_Status']:
        if c in df.columns:
            cat_for_box = c
            break
    if cat_for_box:
        fig, ax = plt.subplots()
        sns.boxplot(x=cat_for_box, y='Price_INR', data=df, ax=ax)
        ax.set_title(f"{cat_for_box} vs Market Price")
        st.pyplot(fig)
