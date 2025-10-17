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
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            except ImportError:
                st.error("❌ openpyxl library is missing. Install it using `pip install openpyxl`.")
                st.stop()
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
    # Price Prediction Form (Dynamic Brand→Model→Other Details)
    # -------------------------------
    st.subheader("💰 Predict Car Price")
    brand_col = 'Brand'
    model_col = 'Model'

    if brand_col in encoders:
        inv_brand = {i: cls for i, cls in enumerate(encoders[brand_col].classes_)}
        selected_brand = st.selectbox(f"{brand_col}", options=list(inv_brand.keys()), format_func=lambda x: inv_brand[x])
    else:
        selected_brand = st.selectbox(f"{brand_col}", options=df[brand_col].unique())

    if model_col in encoders:
        df_original = df.copy()
        for col in [brand_col, model_col, 'Car_Type', 'Fuel_Type']:
            if col in encoders:
                df_original[col] = df[col].map(lambda x: encoders[col].inverse_transform([x])[0])
        models_for_brand = df_original[df_original[brand_col] == inv_brand[selected_brand]][model_col].unique()
        models_for_brand_encoded = [encoders[model_col].transform([m])[0] for m in models_for_brand]
        selected_model = st.selectbox(f"{model_col}", options=models_for_brand_encoded, format_func=lambda x: encoders[model_col].inverse_transform([x])[0])
    else:
        selected_model = st.selectbox(f"{model_col}", options=df[model_col].unique())

    # Automatically set other details based on selected brand+model
    auto_row = df_original[(df_original['Brand'] == inv_brand[selected_brand]) & 
                           (df_original['Model'] == encoders[model_col].inverse_transform([selected_model])[0])].iloc[0]

    st.write("### 🚙 Car Details")
    st.write(auto_row)

    # Prediction inputs
    inputs = {}
    for col in feature_columns:
        if col in ['Brand','Model']:
            inputs[col] = selected_brand if col == 'Brand' else selected_model
        else:
            if col in encoders:
                val = auto_row[col]
                inputs[col] = encoders[col].transform([val])[0]
            else:
                inputs[col] = auto_row[col]

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
        st.table(top_models_df)

        st.subheader("📋 Details of Top 5 Models")
        top_model_names = top_models_df['Model'].tolist()
        top_model_details = df[df['Model'].isin(top_model_names)]
        st.dataframe(top_model_details)

else:
    st.info("📥 Please upload your dataset to start.")
