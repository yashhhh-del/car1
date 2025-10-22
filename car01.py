import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Smart Car Price Predictor", layout="wide")

st.title("🚗 Smart Car Price Prediction App")
st.markdown("### AI-based Car Price Estimator (Based on Your Dataset)")

# ---------------------- Upload Dataset ----------------------
uploaded_file = st.file_uploader("📂 Upload your car dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")
    st.dataframe(df.head())

    # Check for Market_Price column
    if 'Market_Price(INR)' not in df.columns:
        st.warning("🟡 'Market_Price(INR)' column missing — adding dummy values for now.")
        df['Market_Price(INR)'] = np.random.randint(300000, 1500000, df.shape[0])

    # Clean and encode
    df = df.dropna()
    le = LabelEncoder()

    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=['Market_Price(INR)'])
    y = df['Market_Price(INR)']

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Performance
    st.subheader("📊 Model Performance")
    st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    st.write(f"**R2 Score:** {r2_score(y_test, y_pred):.2f}")

    # Brand and Model selection filters
    if 'Brand' in df.columns and 'Model' in df.columns:
        brand_list = df['Brand'].unique().tolist()
        selected_brand = st.selectbox("Select Car Brand", brand_list)

        model_list = df[df['Brand'] == selected_brand]['Model'].unique().tolist()
        selected_model = st.selectbox("Select Car Model", model_list)

        filtered_data = df[(df['Brand'] == selected_brand) & (df['Model'] == selected_model)]

        if not filtered_data.empty:
            st.markdown(f"### 🔍 Showing Predictions for **{selected_brand} {selected_model}**")

            X_filtered = filtered_data.drop(columns=['Market_Price(INR)'])
            X_filtered_scaled = scaler.transform(X_filtered)
            predicted_prices = model.predict(X_filtered_scaled)

            # Show min, mid, max
            st.write(f"**Min Predicted Price:** ₹{predicted_prices.min():,.0f}")
            st.write(f"**Avg Predicted Price:** ₹{predicted_prices.mean():,.0f}")
            st.write(f"**Max Predicted Price:** ₹{predicted_prices.max():,.0f}")

            # Car image via Google
            search_query = f"{selected_brand}+{selected_model}+car"
            image_url = f"https://source.unsplash.com/600x400/?{search_query}"
            st.image(image_url, caption=f"{selected_brand} {selected_model}", use_container_width=True)

            # Plot predicted price distribution
            st.subheader("📈 Predicted Price Distribution")
            fig, ax = plt.subplots()
            sns.histplot(predicted_prices, bins=20, kde=True, ax=ax)
            ax.set_title("Predicted Market Price Distribution")
            ax.set_xlabel("Price (INR)")
            st.pyplot(fig)

        else:
            st.warning("No data found for selected brand and model.")
else:
    st.info("👆 Please upload your CSV file to start.")
