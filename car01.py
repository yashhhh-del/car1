import streamlit as st
import pandas as pd
import numpy as np# ======================================================
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

sns.set(style="whitegrid")
st.set_page_config(page_title="Smart Car Pricing PRO", layout="wide")

# =======================
# MAIN APP FUNCTION
# =======================
def main():
    st.title("🚗 Smart Pricing System for Used Cars - PRO")
    st.markdown("### Upload your dataset and get AI-powered price predictions & market insights!")

    # -------------------------------
    # File Upload with caching
    # -------------------------------
    @st.cache_data
    def load_csv(uploaded_file):
        return pd.read_csv(uploaded_file)

    uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

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
        # Price Prediction Form (Dynamic Brand→Model)
        # -------------------------------
        st.subheader("💰 Predict Car Price")

        brand_col = 'Brand'
        model_col = 'Model'

        # Full list of brands available
        if brand_col in encoders:
            inv_brand = {i: cls for i, cls in enumerate(encoders[brand_col].classes_)}
            all_brands = list(inv_brand.keys())
            selected_brand = st.selectbox(f"{brand_col}", options=all_brands, format_func=lambda x: inv_brand[x])
        else:
            all_brands = df[brand_col].unique()
            selected_brand = st.selectbox(f"{brand_col}", options=all_brands)

        # Filter models dynamically for selected brand
        if model_col in encoders:
            df_original = df.copy()
            # Decode brand & model
            df_original[brand_col] = df[brand_col].map(lambda x: encoders[brand_col].inverse_transform([x])[0])
            df_original[model_col] = df[model_col].map(lambda x: encoders[model_col].inverse_transform([x])[0])

            # Models for selected brand
            models_for_brand = df_original[df_original[brand_col] == inv_brand[selected_brand]][model_col].unique()
            models_for_brand_encoded = [encoders[model_col].transform([m])[0] for m in models_for_brand]
            selected_model = st.selectbox(f"{model_col}", options=models_for_brand_encoded,
                                          format_func=lambda x: encoders[model_col].inverse_transform([x])[0])
        else:
            selected_model = st.selectbox(f"{model_col}", options=df[model_col].unique())

        # Other features dynamically
        inputs = {brand_col: selected_brand, model_col: selected_model}
        for col in feature_columns:
            if col not in [brand_col, model_col]:
                if col in encoders:
                    inv_map = {i: cls for i, cls in enumerate(encoders[col].classes_())}
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

    else:
        st.info("📥 Please upload your dataset to start.")


# =======================
# RUN MAIN
# =======================
if __name__ == "__main__":
    main()

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

