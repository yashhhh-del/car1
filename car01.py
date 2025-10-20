# ======================================================
# SMART CAR PRICING SYSTEM - CLEANED VERSION
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

# Page config
st.set_page_config(page_title="Smart Car Pricing System", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Custom CSS
if st.session_state.dark_mode:
    st.markdown("""
    <style>
        .main {background-color: #1e1e1e; color: #ffffff;}
        .stApp {background-color: #1e1e1e;}
        h1, h2, h3 {color: #4A90E2 !important;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem 0;
        }
        .stMetric {
            background-color: #e0f7fa;
            border-left: 5px solid #00bcd4;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚗 Smart Car Pricing System</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Price Prediction")

sns.set(style="whitegrid")

# Helper Functions
def calculate_depreciation(price, year, current_year=2024):
    age = current_year - year
    depreciation_rate = 0.15
    current_value = price * ((1 - depreciation_rate) ** age)
    future_values = []
    for i in range(1, 6):
        future_value = current_value * ((1 - depreciation_rate) ** i)
        future_values.append({'Year': current_year + i, 'Value': future_value})
    return current_value, future_values

def calculate_emi(principal, rate, tenure_months):
    """Calculate EMI"""
    if principal <= 0:
        return 0
    monthly_rate = rate / (12 * 100)
    emi = principal * monthly_rate * ((1 + monthly_rate)**tenure_months) / (((1 + monthly_rate)**tenure_months) - 1)
    return emi

def simple_chatbot(query, df):
    query = query.lower()
    if 'cheap' in query or 'budget' in query or 'under' in query:
        try:
            import re
            price_match = re.search(r'(\d+)\s*(lakhs|lacs|million)?', query)
            if price_match:
                price_value = int(price_match.group(1))
                if price_match.group(2) in ['lakhs', 'lacs']:
                    price = price_value * 100000
                elif price_match.group(2) == 'million':
                    price = price_value * 1000000
                else:
                    price = price_value
            else:
                price = None
                
            if price:
                cars = df[df['Market_Price(INR)'] <= price].nsmallest(5, 'Market_Price(INR)')
                return f"Found {len(cars)} cars under ₹{price:,}", cars
            else:
                cars = df.nsmallest(5, 'Market_Price(INR)')
                return "Could not understand specific price. Here are the 5 cheapest cars:", cars
        except:
            cars = df.nsmallest(5, 'Market_Price(INR)')
            return "Here are the 5 cheapest cars:", cars
    elif 'suv' in query:
        cars = df[df['Car_Type'].str.contains('SUV', case=False, na=False)].head(5)
        return "Top 5 SUVs:", cars
    elif 'sedan' in query:
        cars = df[df['Car_Type'].str.contains('Sedan', case=False, na=False)].head(5)
        return "Top 5 Sedans:", cars
    elif 'latest' in query or 'new' in query:
        cars = df.nlargest(5, 'Year')
        return "Latest cars:", cars
    elif 'highest mileage' in query or 'most fuel efficient' in query:
        if 'Mileage(kmpl)' in df.columns:
            cars = df.nlargest(5, 'Mileage(kmpl)')[['Brand', 'Model', 'Mileage(kmpl)', 'Market_Price(INR)']]
            return "Here are 5 cars with the highest mileage:", cars
        else:
            return "Mileage data not available in the dataset.", pd.DataFrame()
    else:
        return "Try: 'cars under 10 lakhs', 'best SUV', 'latest cars', 'highest mileage'", pd.DataFrame()

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    dark_mode = st.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    st.markdown("---")
    
    st.title("📊 Navigation")
    page = st.radio("Go to", [
        "🏠 Home",
        "💰 Price Prediction",
        "📊 Compare Cars",
        "🧮 EMI Calculator"
    ])

# File Upload
uploaded_file = st.file_uploader("📂 Upload CSV/XLSX File", type=["csv","xlsx"])

df = None
feature_columns = []

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            import openpyxl
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.success("✅ File uploaded!")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

    if df is not None:
        if 'Market_Price(INR)' not in df.columns:
            st.error("❌ Must include 'Market_Price(INR)' column")
            st.stop()

        # Data Preprocessing
        df_clean = df.dropna()
        if 'Year' in df_clean.columns:
            df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
            df_clean = df_clean.dropna(subset=['Year'])
            df_clean['Year'] = df_clean['Year'].astype(int)

        cat_cols = df_clean.select_dtypes(include=['object']).columns
        encoders = {}
        df_encoded = df_clean.copy()
        
        for col in cat_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_clean[col].astype(str))
            encoders[col] = le

        # Model Training
        cols_to_drop = ['Market_Price(INR)'] + [col for col in df_encoded.columns if df_encoded[col].dtype == 'object']
        cols_to_drop = list(set(cols_to_drop))

        X = df_encoded.drop(columns=[col for col in cols_to_drop if col in df_encoded.columns], errors='ignore')
        y = df_encoded['Market_Price(INR)']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        results = {}
        trained_models = {}
        best_model = None

        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
            y_pred = model.predict(X_test)
            results[name] = {
                'R2 Score': r2_score(y_test, y_pred)
            }

        result_df = pd.DataFrame(results).T
        if not result_df.empty:
            best_model_name = result_df['R2 Score'].idxmax()
            best_model = trained_models[best_model_name]
        else:
            st.error("No models could be trained. Please check your data.")
            st.stop()
    else:
        st.stop()

# ============================================
# HOME PAGE
# ============================================
if page == "🏠 Home":
    if df is not None:
        st.subheader("📊 Market Insights Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cars", f"{len(df_clean):,}")
        with col2:
            st.metric("Unique Brands", f"{df_clean['Brand'].nunique()}")
        with col3:
            st.metric("Avg. Price", f"₹{df_clean['Market_Price(INR)'].mean()/100000:.1f} Lakhs")
        with col4:
            st.metric("Predictions Made", len(st.session_state.predictions))

        st.markdown("---")
        
        # Price Distribution by Brand
        if 'Brand' in df_clean.columns:
            st.markdown("#### Price Distribution by Brand")
            selected_brands = st.multiselect(
                "Select Brands", 
                options=sorted(df_clean['Brand'].unique()),
                default=sorted(df_clean['Brand'].unique())[:5]
            )
            
            if selected_brands:
                filtered_df = df_clean[df_clean['Brand'].isin(selected_brands)]
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.boxplot(x='Brand', y='Market_Price(INR)', data=filtered_df, ax=ax, palette='coolwarm')
                ax.set_title('Market Price Distribution by Brand')
                ax.set_ylabel('Market Price (INR)')
                ax.ticklabel_format(style='plain', axis='y')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("---")
        st.markdown("### 💬 Ask About Cars")
        user_query = st.text_input("e.g., 'cars under 10 lakhs', 'best SUV', 'highest mileage'")
        if user_query:
            response, cars_found = simple_chatbot(user_query, df_clean)
            st.write(response)
            if not cars_found.empty:
                st.dataframe(cars_found, use_container_width=True)
    else:
        st.info("📥 Upload a CSV/XLSX file to get started!")

# ============================================
# PRICE PREDICTION
# ============================================
elif page == "💰 Price Prediction":
    if df is not None:
        st.subheader("💰 AI Price Prediction")
        
        brands = sorted(df_clean['Brand'].unique())
        selected_brand = st.selectbox("Brand", brands)
        
        filtered_models = sorted(df_clean[df_clean['Brand'] == selected_brand]['Model'].unique())
        selected_model = st.selectbox("Model", filtered_models)
        
        filtered_rows = df_clean[(df_clean['Brand'] == selected_brand) & (df_clean['Model'] == selected_model)]
        
        if len(filtered_rows) > 0:
            filtered_row = filtered_rows.iloc[0]
            
            col1, col2, col3 = st.columns(3)
            inputs = {}
            
            feature_idx = 0
            for col in feature_columns:
                if col in filtered_row.index:
                    with [col1, col2, col3][feature_idx % 3]:
                        if col in encoders:
                            original_values = df_clean[col].unique()
                            options = sorted(original_values)
                            default_val = df_clean[df_clean['Brand'] == selected_brand][col].iloc[0]
                            inputs[col] = st.selectbox(f"{col}", options, index=list(options).index(default_val) if default_val in options else 0, key=f"p_{col}")
                        else:
                            min_val = int(df_clean[col].min())
                            max_val = int(df_clean[col].max())
                            default_val = int(filtered_row[col])
                            inputs[col] = st.slider(f"{col}", min_val, max_val, default_val, key=f"p_{col}")
                    feature_idx += 1
            
            if st.button("🔍 Predict Price", type="primary"):
                predict_input_data = {}
                for col in feature_columns:
                    if col in inputs:
                        predict_input_data[col] = inputs[col]
                    elif col in filtered_row.index:
                        predict_input_data[col] = filtered_row[col]
                    else:
                        predict_input_data[col] = 0
                
                input_df = pd.DataFrame([predict_input_data])
                
                for col in encoders:
                    if col in input_df.columns and input_df[col].dtype == 'object':
                        try:
                            input_df[col] = encoders[col].transform(input_df[col].astype(str))
                        except ValueError:
                            input_df[col] = encoders[col].transform([encoders[col].classes_[0]])
                
                missing_cols = set(X.columns) - set(input_df.columns)
                for c in missing_cols:
                    input_df[c] = 0
                input_df = input_df[X.columns]

                input_scaled = scaler.transform(input_df)
                predicted_price = best_model.predict(input_scaled)[0]

                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min Price", f"₹{predicted_price*0.9:,.0f}")
                with col2:
                    st.metric("Fair Price", f"₹{predicted_price:,.0f}")
                with col3:
                    st.metric("Max Price", f"₹{predicted_price*1.1:,.0f}")
                
                st.balloons()
                
                st.session_state.predictions.append({
                    'Brand': selected_brand,
                    'Model': selected_model,
                    'Price': f"₹{predicted_price:,.0f}",
                    'Time': datetime.now().strftime("%Y-%m-%d %H:%M")
                })
        else:
            st.warning("No data found for selected car.")
    else:
        st.info("📥 Upload a file to make predictions!")

# ============================================
# COMPARE CARS
# ============================================
elif page == "📊 Compare Cars":
    if df is not None:
        st.subheader("📊 Compare Cars Side by Side")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Car 1")
            brands1 = sorted(df_clean['Brand'].unique())
            brand1 = st.selectbox("Brand", brands1, key="brand1")
            models1 = sorted(df_clean[df_clean['Brand'] == brand1]['Model'].unique())
            model1 = st.selectbox("Model", models1, key="model1")
            car1_data = df_clean[(df_clean['Brand'] == brand1) & (df_clean['Model'] == model1)].iloc[0]
        
        with col2:
            st.markdown("### Car 2")
            brands2 = sorted(df_clean['Brand'].unique())
            brand2 = st.selectbox("Brand", brands2, key="brand2")
            models2 = sorted(df_clean[df_clean['Brand'] == brand2]['Model'].unique())
            model2 = st.selectbox("Model", models2, key="model2")
            car2_data = df_clean[(df_clean['Brand'] == brand2) & (df_clean['Model'] == model2)].iloc[0]
        
        if st.button("Compare", type="primary"):
            st.markdown("---")
            st.subheader("Comparison Results")
            
            comparison_data = {
                'Feature': ['Brand', 'Model', 'Price', 'Year'],
                'Car 1': [brand1, model1, f"₹{car1_data['Market_Price(INR)']:,.0f}", car1_data['Year']],
                'Car 2': [brand2, model2, f"₹{car2_data['Market_Price(INR)']:,.0f}", car2_data['Year']]
            }
            
            if 'Mileage(kmpl)' in df_clean.columns:
                comparison_data['Feature'].append('Mileage')
                comparison_data['Car 1'].append(f"{car1_data.get('Mileage(kmpl)', 'N/A')} kmpl")
                comparison_data['Car 2'].append(f"{car2_data.get('Mileage(kmpl)', 'N/A')} kmpl")
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Price comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))
            cars = [f"{brand1} {model1}", f"{brand2} {model2}"]
            prices = [car1_data['Market_Price(INR)'], car2_data['Market_Price(INR)']]]
            ax.bar(cars, prices, color=['#667eea', '#764ba2'])
            ax.set_ylabel('Price (INR)')
            ax.set_title('Price Comparison')
            ax.ticklabel_format(style='plain', axis='y')
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("📥 Upload a file to compare cars!")

# ============================================
# EMI CALCULATOR
# ============================================
elif page == "🧮 EMI Calculator":
    st.subheader("🧮 EMI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        car_price = st.number_input("Car Price (₹)", 100000, 10000000, 1000000, step=50000)
        down_payment = st.slider("Down Payment (%)", 0, 50, 20)
        interest_rate = st.slider("Interest Rate (% per year)", 5.0, 15.0, 9.5, 0.1)
    
    with col2:
        tenure_years = st.slider("Loan Tenure (years)", 1, 7, 5)
        
    loan_amount = car_price * (1 - down_payment/100)
    tenure_months = tenure_years * 12
    
    emi = calculate_emi(loan_amount, interest_rate, tenure_months)
    total_payment = emi * tenure_months
    total_interest = total_payment - loan_amount
    
    st.markdown("---")
    st.subheader("💰 EMI Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Monthly EMI", f"₹{emi:,.0f}")
    with col2:
        st.metric("Loan Amount", f"₹{loan_amount:,.0f}")
    with col3:
        st.metric("Total Interest", f"₹{total_interest:,.0f}")
    with col4:
        st.metric("Total Payment", f"₹{total_payment:,.0f}")
    
    # Payment breakdown chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie([loan_amount, total_interest], labels=['Principal', 'Interest'], 
           autopct='%1.1f%%', startangle=90, colors=['#667eea', '#764ba2'])
    ax.set_title('Loan Payment Breakdown')
    st.pyplot(fig)
    plt.close(fig)

# Footer
if uploaded_file is None:
    st.info("📥 Upload a CSV/XLSX file to get started!")
