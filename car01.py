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
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")
        st.stop()

    st.subheader("📊 Full Dataset")
    st.dataframe(df)

    # -------------------------------
    # Data Preprocessing
    # -------------------------------
    st.subheader("🧹 Data Cleaning & Preprocessing")
    df = df.dropna()

    cat_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # -------------------------------
    # Model Training
    # -------------------------------
    st.subheader("🤖 Model Training & Evaluation")
    if 'Market_Price(INR)' not in df.columns:
        st.error("❌ Dataset must have 'Market_Price(INR)' column for prediction.")
        st.stop()

    X = df.drop(columns=['Market_Price(INR)'], errors='ignore')
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
    st.subheader("📈 Model Performance Comparison")
    result_df = pd.DataFrame(results).T
    st.dataframe(result_df)

    best_model_name = result_df['R2 Score'].idxmax()
    best_model = models[best_model_name]
    st.success(f"🏆 Best Model Selected: **{best_model_name}**")

    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        st.subheader("🌟 Feature Importance")
        importance = best_model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
        st.bar_chart(fi_df.set_index('Feature'))

    # -------------------------------
    # Price Prediction Form (Dynamic Brand → Model → Car_Type → Fuel_Type)
    # -------------------------------
    st.subheader("💰 Predict Car Price")

    # Brand select
    selected_brand = st.selectbox("Select Brand", options=[encoders['Brand'].inverse_transform([b])[0] for b in df['Brand'].unique()])

    # Filter models based on selected brand
    brand_filtered = df[df['Brand'] == encoders['Brand'].transform([selected_brand])[0]]
    models_for_brand = [encoders['Model'].inverse_transform([m])[0] for m in brand_filtered['Model'].unique()]
    selected_model = st.selectbox("Select Model", options=models_for_brand)

    # Filter car types based on selected brand and model
    model_filtered = brand_filtered[brand_filtered['Model'] == encoders['Model'].transform([selected_model])[0]]
    car_types_for_model = [encoders['Car_Type'].inverse_transform([c])[0] for c in model_filtered['Car_Type'].unique()]
    selected_car_type = st.selectbox("Select Car Type", options=car_types_for_model)

    # Filter fuel types based on brand, model, car type
    car_filtered = model_filtered[model_filtered['Car_Type'] == encoders['Car_Type'].transform([selected_car_type])[0]]
    fuel_types_for_car = [encoders['Fuel_Type'].inverse_transform([f])[0] for f in car_filtered['Fuel_Type'].unique()]
    selected_fuel_type = st.selectbox("Select Fuel Type", options=fuel_types_for_car)

    # -------------------------------
    # Other feature inputs
    # -------------------------------
    inputs = {
        'Brand': encoders['Brand'].transform([selected_brand])[0],
        'Model': encoders['Model'].transform([selected_model])[0],
        'Car_Type': encoders['Car_Type'].transform([selected_car_type])[0],
        'Fuel_Type': encoders['Fuel_Type'].transform([selected_fuel_type])[0]
    }

    for col in feature_columns:
        if col not in ['Brand','Model','Car_Type','Fuel_Type']:
            if col in encoders:
                inv_map = {i: cls for i, cls in enumerate(encoders[col].classes_)}
                inputs[col] = st.selectbox(f"{col}", options=list(inv_map.keys()), format_func=lambda x: inv_map[x])
            else:
                min_val = int(df[col].min())
                max_val = int(df[col].max())
                default_val = int(df[col].median())
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
        sns.histplot(df['Market_Price(INR)'], kde=True, ax=ax)
        ax.set_title("Distribution of Market Prices")
        st.pyplot(fig)

    with col2:
        cat_for_box = None
        for c in ['Fuel_Type', 'Transmission', 'Owner']:
            if c in df.columns:
                cat_for_box = c
                break
        if cat_for_box:
            fig, ax = plt.subplots()
            sns.boxplot(x=cat_for_box, y='Market_Price(INR)', data=df, ax=ax)
            ax.set_title(f"{cat_for_box} vs Market Price")
            st.pyplot(fig)

    # -------------------------------
    # Top 5 Models Analysis
    # -------------------------------
    st.subheader("🏆 Top 5 Models by Average Market Price")
    if 'Model' in df.columns and 'Market_Price(INR)' in df.columns:
        top_models_df = df.groupby('Model')['Market_Price(INR)'].mean().sort_values(ascending=False).head(5).reset_index()
        top_models_df['Model'] = [encoders['Model'].inverse_transform([m])[0] for m in top_models_df['Model']]
        st.table(top_models_df)

        st.subheader("📋 Details of Top 5 Models")
        top_model_names = [encoders['Model'].transform([m])[0] for m in top_models_df['Model']]
        top_model_details = df[df['Model'].isin(top_model_names)]
        for col in ['Brand','Model','Car_Type','Fuel_Type']:
