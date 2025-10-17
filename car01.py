# ===========================================================
# Smart Pricing System for Used Cars - Enhanced Streamlit App
# ===========================================================

import os
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
import streamlit as st

sns.set(style="whitegrid")

# =========================
# 1. Load Dataset
# =========================
@st.cache_data
def load_data():
    file_path = "./Smart_Car_Pricing_4000.csv"
    if not os.path.exists(file_path):
        st.error(f"CSV file not found at {file_path}")
        st.stop()
    df = pd.read_csv(file_path)
    return df

# =========================
# 2. Data Preprocessing
# =========================
def preprocess_data(df):
    df = df.dropna()

    label_encoders = {}
    categorical_cols = ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Owner', 'Location']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    scaler = StandardScaler()
    numerical_cols = ['Age', 'Mileage(km)']
    if all(col in df.columns for col in numerical_cols):
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, label_encoders, scaler

# =========================
# 3. Model Training
# =========================
def train_model(df):
    X = df.drop(columns=[col for col in ['Car_ID', 'Market_Price(INR)'] if col in df.columns])
    y = df['Market_Price(INR)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_score = float('inf')

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        if mae < best_score:
            best_score = mae
            best_model = model
            best_model_name = name

    y_pred = best_model.predict(X_test)
    st.subheader("📊 Model Performance Metrics")
    st.write(f"**Best Model:** {best_model_name}")
    st.write("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
    st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
    st.write("R² Score:", round(r2_score(y_test, y_pred), 3))

    with open("car_price_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return best_model

# =========================
# 4. Prediction + Negotiation Range
# =========================
def predict_price(model, input_data):
    predicted = model.predict(input_data)[0]
    min_price = predicted * 0.9
    max_price = predicted * 1.1
    return predicted, min_price, max_price

# =========================
# 5. Streamlit App
# =========================
def main():
    st.title("🚗 Smart Pricing System for Used Cars - Enhanced Dashboard")

    # Load Dataset
    df = load_data()
    st.subheader("📂 Full Dataset Preview")
    st.dataframe(df)

    # Preprocess Data
    df_processed, le_dict, scaler = preprocess_data(df)

    # Train Model
    st.subheader("🤖 Model Training Progress")
    model = train_model(df_processed)

    # Prediction Example
    st.subheader("💰 Prediction Example (Demo)")
    input_example = df_processed.drop(columns=[col for col in ['Car_ID', 'Market_Price(INR)'] if col in df_processed.columns]).iloc[0:1]
    predicted, min_p, max_p = predict_price(model, input_example)

    st.success(f"**Predicted Fair Price:** ₹{predicted:,.0f}")
    st.info(f"Negotiation Range → Min: ₹{min_p:,.0f} | Max: ₹{max_p:,.0f}")

    # =======================
    # Visualizations Section
    # =======================
    st.subheader("📉 Visual Insights")

    # Price Distribution
    if 'Market_Price(INR)' in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df['Market_Price(INR)'], kde=True, color="skyblue", ax=ax)
        ax.set_title("Market Price Distribution")
        st.pyplot(fig)

    # Brand-wise Average Price
    if 'Brand' in df.columns and 'Market_Price(INR)' in df.columns:
        brand_avg = df.groupby('Brand')['Market_Price(INR)'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=brand_avg.values, y=brand_avg.index, palette="viridis", ax=ax)
        ax.set_title("Top 10 Brands by Average Resale Price")
        st.pyplot(fig)

    # Depreciation Trend (Year vs Price)
    if 'Year' in df.columns and 'Market_Price(INR)' in df.columns:
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x='Year', y='Market_Price(INR)', marker="o", ax=ax)
        ax.set_title("Depreciation Trend: Year vs Market Price")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
