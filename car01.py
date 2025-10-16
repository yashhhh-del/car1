# ===============================================
# Smart Pricing System for Used Cars (Streamlit)
# ===============================================

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
    file_path = "./Smart_Car_Pricing_4000.csv"  # Relative path
    if not os.path.exists(file_path):
        st.error(f"CSV file not found at {file_path}")
        st.stop()
    df = pd.read_csv(file_path)
    return df

# =========================
# 2. Data Preprocessing (Robust)
# =========================
def preprocess_data(df):
    df = df.dropna()

    # Debug: show all columns
    st.write("Columns in dataset:", df.columns.tolist())

    # Encode categorical columns safely
    label_encoders = {}
    categorical_cols = ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Owner', 'Location']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Scale numerical columns safely
    scaler = StandardScaler()
    numerical_cols = ['Age', 'Mileage(km)']
    numerical_cols = [col for col in numerical_cols if col in df.columns]  # Only existing columns
    if numerical_cols:  # Avoid empty list
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, label_encoders, scaler

# =========================
# 3. Train Model
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

    # Evaluate
    y_pred = best_model.predict(X_test)
    st.subheader("Model Performance")
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    st.write("R2 Score:", r2_score(y_test, y_pred))

    # Save model
    with open("car_price_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return best_model

# =========================
# 4. Prediction
# =========================
def predict_price(model, input_data):
    return model.predict(input_data)

# =========================
# 5. Streamlit App
# =========================
def main():
    st.title("Smart Pricing System for Used Cars 🚗")

    # Load Dataset
    df = load_data()
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Preprocess Data
    df_processed, le_dict, scaler = preprocess_data(df)
    st.subheader("Data after Preprocessing")
    st.dataframe(df_processed.head())

    # Train Model
    st.subheader("Training Model...")
    model = train_model(df_processed)

    # Prediction Example
    st.subheader("Prediction Example")
    input_example = df_processed.drop(columns=[col for col in ['Car_ID', 'Market_Price(INR)'] if col in df_processed.columns]).iloc[0:1]
    predicted_price = predict_price(model, input_example)
    st.write("Predicted Market Price for first car:", predicted_price[0])

    # Visualization: Price Distribution
    st.subheader("Market Price Distribution")
    if 'Market_Price(INR)' in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df['Market_Price(INR)'], kde=True, ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
