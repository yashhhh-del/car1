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

# -----------------------------------------
st.set_page_config(page_title="Smart Car Pricing PRO", layout="wide")
st.title("🚗 Smart Pricing System for Used Cars - PRO")
st.markdown("### Upload your dataset and get AI-powered price predictions & market insights!")
sns.set(style="whitegrid")

# -----------------------------------------
# File Upload
# -----------------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV or Excel File", type=["csv", "xlsx"])

if uploaded_file is not None:
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

    # -----------------------------------------
    # Data Cleaning & Encoding
    # -----------------------------------------
    st.subheader("🧹 Data Cleaning & Preprocessing")
    df = df.dropna()
    cat_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # -----------------------------------------
    # Model Training
    # -----------------------------------------
    st.subheader("🤖 Model Training & Evaluation")
    if 'Market_Price(INR)' not in df.columns:
        st.error("❌ Dataset must have 'Market_Price(INR)' column for prediction.")
        st.stop()

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
    st.subheader("📈 Model Performance Comparison")
    result_df = pd.DataFrame(results).T
    st.dataframe(result_df)

    best_model_name = result_df['R2 Score'].idxmax()
    best_model = models[best_model_name]
    st.success(f"🏆 Best Model Selected: **{best_model_name}**")

    # -----------------------------------------
    # Dynamic Brand → Model → Type → Fuel
    # -----------------------------------------
    st.subheader("💰 Predict Car Price")
    df_original = df.copy()
    for col in encoders:
        df_original[col] = encoders[col].inverse_transform(df[col])

    brand_list = df_original['Brand'].unique()
    selected_brand = st.selectbox("Select Brand", brand_list)

    models_for_brand = df_original[df_original['Brand'] == selected_brand]['Model'].unique()
    selected_model = st.selectbox("Select Model", models_for_brand)

    filtered_data = df_original[
        (df_original['Brand'] == selected_brand) & (df_original['Model'] == selected_model)
    ].head(1)

    if not filtered_data.empty:
        st.subheader("📋 Car Details (Auto-filled)")
        for col in ['Brand', 'Model', 'Car_Type', 'Year', 'Age', 'Fuel_Type', 'Transmission',
                    'Mileage(km)', 'Engine_cc', 'Power_HP', 'Seats', 'Market_Price(INR)',
                    'Condition', 'Owner', 'Insurance_Status', 'Registration_City',
                    'Service_History', 'Accident_History', 'Car_Availability']:
            if col in filtered_data.columns:
                st.write(f"**{col}:** {filtered_data.iloc[0][col]}")

    # -----------------------------------------
    # Prediction Section
    # -----------------------------------------
    if st.button("🔍 Predict Price"):
        input_data = filtered_data.drop(columns=['Market_Price(INR)'], errors='ignore')
        input_encoded = input_data.copy()
        for col in encoders:
            input_encoded[col] = encoders[col].transform(input_encoded[col].astype(str))

        input_scaled = scaler.transform(input_encoded)
        predicted_price = best_model.predict(input_scaled)[0]

        st.subheader("📊 Price Estimation")
        st.metric("Minimum Negotiation Price", f"₹{predicted_price*0.9:,.0f}")
        st.metric("Fair Market Price", f"₹{predicted_price:,.0f}")
        st.metric("Maximum Negotiation Price", f"₹{predicted_price*1.1:,.0f}")

    # -----------------------------------------
    # Market Insights
    # -----------------------------------------
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

    # -----------------------------------------
    # Top 5 Models by Price
    # -----------------------------------------
    st.subheader("🏆 Top 5 Models by Average Market Price")
    top_models_df = (
        df_original.groupby('Model')['Market_Price(INR)']
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    st.table(top_models_df)

    st.subheader("📋 Details of Top 5 Models")
    for _, row in top_models_df.iterrows():
        model_name = row['Model']
        st.markdown(f"### 🚘 {model_name}")
        model_details = df_original[df_original['Model'] == model_name].iloc[0]
        for col in ['Brand', 'Model', 'Car_Type', 'Year', 'Fuel_Type', 'Transmission',
                    'Mileage(km)', 'Power_HP', 'Seats', 'Market_Price(INR)']:
            if col in model_details.index:
                st.write(f"**{col}**: {model_details[col]}")

else:
    st.info("📥 Please upload your dataset to start.")
