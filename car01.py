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
st.title("🚗 Smart Pricing System for Used Cars - PRO")
st.markdown("### Upload your dataset and get AI-powered price predictions & market insights!")
sns.set(style="whitegrid")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV or Excel File", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load file
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            import openpyxl
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")
        st.stop()

    st.subheader("📊 Full Dataset")
    st.dataframe(df)

    # -------------------------------
    # Data Cleaning & Encoding
    # -------------------------------
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
    if 'Market_Price(INR)' not in df.columns:
        st.error("❌ Dataset must have 'Market_Price(INR)' column for prediction.")
        st.stop()

    X = df.drop(columns=['Market_Price(INR)'])
    y = df['Market_Price(INR)']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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

    # -------------------------------
    # Show Model Performance
    # -------------------------------
    st.subheader("📈 Model Performance")
    result_df = pd.DataFrame(results).T
    st.dataframe(result_df)

    best_model_name = result_df['R2 Score'].idxmax()
    best_model = models[best_model_name]
    st.success(f"🏆 Best Model: {best_model_name}")

    st.markdown("**Prediction Example for first car in dataset:**")
    first_car = X.iloc[0]
    first_car_scaled = scaler.transform([first_car])
    predicted_price = best_model.predict(first_car_scaled)[0]
    st.metric("Predicted Market Price", f"₹{predicted_price:,.0f}")

    # -------------------------------
    # Market Insights
    # -------------------------------
    df_original = df.copy()
    for col in encoders:
        df_original[col] = encoders[col].inverse_transform(df[col])

    st.subheader("📉 Market Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_original['Market_Price(INR)'], kde=True, ax=ax, color='skyblue')
    ax.set_title("Market Price Distribution")
    st.pyplot(fig)

    if 'Fuel_Type' in df_original.columns:
        st.subheader("⛽ Fuel Type vs Price")
        fig, ax = plt.subplots()
        sns.boxplot(x='Fuel_Type', y='Market_Price(INR)', data=df_original, ax=ax)
        ax.set_title("Fuel Type vs Market Price")
        st.pyplot(fig)

    # -------------------------------
    # Top 5 Models
    # -------------------------------
    st.subheader("🏆 Top 5 Models by Average Price")
    top_models = df_original.groupby('Model')['Market_Price(INR)'].mean().sort_values(ascending=False).head(5)
    st.table(top_models.reset_index())
