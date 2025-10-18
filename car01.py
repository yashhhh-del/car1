# ======================================================
# SMART PRICING SYSTEM FOR USED CARS - ULTIMATE VERSION
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
from datetime import datetime, timedelta
import json

# Page config
st.set_page_config(page_title="Smart Car Pricing ULTIMATE", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'wishlist' not in st.session_state:
    st.session_state.wishlist = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'reviews' not in st.session_state:
    st.session_state.reviews = {}
if 'price_alerts' not in st.session_state:
    st.session_state.price_alerts = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Custom CSS
if st.session_state.dark_mode:
    st.markdown("""
    <style>
        .main {background-color: #1e1e1e; color: #ffffff;}
        .stApp {background-color: #1e1e1e;}
        h1, h2, h3 {color: #4A90E2 !important;}
        .main-header {color: #4A90E2 !important;}
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
        .feature-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚗 Smart Car Pricing System ULTIMATE</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Price Predictions | EMI Calculator | Depreciation Analyzer | & More!")

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

def calculate_fuel_cost(mileage_per_km, fuel_type, yearly_km=15000):
    fuel_prices = {'Petrol': 105, 'Diesel': 95, 'CNG': 80, 'Electric': 8, 'Hybrid': 90}
    price_per_unit = fuel_prices.get(fuel_type, 100)
    if fuel_type == 'Electric':
        yearly_cost = (yearly_km / mileage_per_km) * price_per_unit
    else:
        yearly_cost = (yearly_km / mileage_per_km) * price_per_unit
    return yearly_cost

def calculate_total_ownership_cost(car_price, fuel_cost, insurance_percent=0.03, maintenance=50000):
    insurance = car_price * insurance_percent
    total_yearly = fuel_cost + insurance + maintenance
    total_5_years = total_yearly * 5
    return total_yearly, total_5_years

def calculate_insurance(car_price, age, city_tier=1):
    base_rate = 0.03
    age_factor = 1 + (age * 0.02)
    city_factor = 1 + (city_tier * 0.01)
    insurance = car_price * base_rate * age_factor * city_factor
    return insurance

def simple_chatbot(query, df):
    query = query.lower()
    if 'cheap' in query or 'budget' in query or 'under' in query:
        try:
            price = int(''.join(filter(str.isdigit, query)))
            cars = df[df['Market_Price(INR)'] <= price].nsmallest(5, 'Market_Price(INR)')
            return f"Found {len(cars)} cars under ₹{price:,}", cars
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
    else:
        return "Try asking: 'cars under 10 lakhs', 'best SUV', 'latest cars'", pd.DataFrame()

# Sidebar
with st.sidebar:
    st.title("🔐 User Panel")
    
    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["Login", "Signup"])
        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome {username}!")
                st.rerun()
        with tab2:
            new_user = st.text_input("Username", key="signup_user")
            new_pass = st.text_input("Password", type="password", key="signup_pass")
            email = st.text_input("Email")
            if st.button("Sign Up"):
                st.session_state.logged_in = True
                st.session_state.username = new_user
                st.success(f"Account created! Welcome {new_user}!")
                st.rerun()
    else:
        st.success(f"👋 Hello, {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
    
    st.markdown("---")
    
    # Settings
    st.title("⚙️ Settings")
    dark_mode = st.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    language = st.selectbox("🌐 Language", ["English", "हिंदी", "मराठी"])
    
    st.markdown("---")
    
    # Navigation
    st.title("📊 Navigation")
    page = st.radio("Go to", [
        "🏠 Home",
        "💰 Price Prediction",
        "📊 Compare Cars",
        "🧮 EMI Calculator",
        "📉 Depreciation Analyzer",
        "⛽ Fuel Cost Calculator",
        "💎 Total Ownership Cost",
        "🤖 AI Chatbot",
        "⭐ Reviews & Ratings",
        "❤️ Wishlist",
        "🔔 Price Alerts",
        "📈 Market Insights",
        "📥 Download Report"
    ])

# File Upload
uploaded_file = st.file_uploader("📂 Upload CSV/XLSX File", type=["csv","xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            import openpyxl
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    if 'Market_Price(INR)' not in df.columns:
        st.error("❌ Dataset must include 'Market_Price(INR)' column.")
        st.stop()

    # Data Preprocessing
    df_clean = df.dropna()
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    encoders = {}
    df_encoded = df_clean.copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_clean[col].astype(str))
        encoders[col] = le

    # Model Training
    X = df_encoded.drop(columns=['Market_Price(INR)'])
    y = df_encoded['Market_Price(INR)']

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
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2 Score': r2_score(y_test, y_pred)
        }

    result_df = pd.DataFrame(results).T
    best_model_name = result_df['R2 Score'].idxmax()
    best_model = trained_models[best_model_name]

    # ============================================
    # HOME PAGE
    # ============================================
    if page == "🏠 Home":
        st.subheader("📊 Dashboard Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Cars", f"{len(df_clean):,}")
        with col2:
            st.metric("Brands", f"{df_clean['Brand'].nunique()}")
        with col3:
            st.metric("Avg Price", f"₹{df_clean['Market_Price(INR)'].mean()/100000:.1f}L")
        with col4:
            st.metric("Wishlist", len(st.session_state.wishlist))
        with col5:
            st.metric("Predictions", len(st.session_state.predictions))

        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 10 Brands")
            brand_counts = df_clean['Brand'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=brand_counts.values, y=brand_counts.index, palette='viridis', ax=ax)
            ax.set_xlabel('Number of Cars')
            st.pyplot(fig)
        
        with col2:
            st.subheader("💎 Top 10 Expensive Cars")
            top_expensive = df_clean.nlargest(10, 'Market_Price(INR)')[['Brand', 'Model', 'Market_Price(INR)', 'Year']]
            top_expensive['Price'] = top_expensive['Market_Price(INR)'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(top_expensive[['Brand', 'Model', 'Year', 'Price']], use_container_width=True, hide_index=True)

        st.markdown("---")
        
        # Recent Search History
        if st.session_state.search_history:
            st.subheader("🕒 Recent Searches")
            recent = pd.DataFrame(st.session_state.search_history[-5:])
            st.dataframe(recent, use_container_width=True, hide_index=True)

        # Quick Search
        st.subheader("🔍 Quick Search")
        col1, col2, col3 = st.columns(3)
        with col1:
            search_brand = st.multiselect("Brand", ["All"] + sorted(df_clean['Brand'].unique().tolist()), key="home_brand")
        with col2:
            if 'Fuel_Type' in df_clean.columns:
                search_fuel = st.multiselect("Fuel Type", ["All"] + sorted(df_clean['Fuel_Type'].unique().tolist()), key="home_fuel")
            else:
                search_fuel = ["All"]
        with col3:
            price_range = st.slider("Price Range (Lakhs)", 0, int(df_clean['Market_Price(INR)'].max()/100000), (0, 50), key="home_price")
        
        filtered_data = df_clean.copy()
        if search_brand and "All" not in search_brand:
            filtered_data = filtered_data[filtered_data['Brand'].isin(search_brand)]
        if search_fuel and "All" not in search_fuel and 'Fuel_Type' in df_clean.columns:
            filtered_data = filtered_data[filtered_data['Fuel_Type'].isin(search_fuel)]
        filtered_data = filtered_data[(filtered_data['Market_Price(INR)'] >= price_range[0]*100000) & 
                                     (filtered_data['Market_Price(INR)'] <= price_range[1]*100000)]
        
        st.write(f"Found {len(filtered_data)} cars")
        st.dataframe(filtered_data.head(20), use_container_width=True)
        
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "search_results.csv", "text/csv")

    # ============================================
    # PRICE PREDICTION PAGE
    # ============================================
    elif page == "💰 Price Prediction":
        st.subheader("💰 AI-Powered Price Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🤖 Model Performance")
            st.dataframe(result_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
            st.success(f"🏆 Best Model: **{best_model_name}**")
        
        with col2:
            st.markdown("### 📊 Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            models_list = list(results.keys())
            r2_scores = [results[m]['R2 Score'] for m in models_list]
            sns.barplot(x=r2_scores, y=models_list, palette='coolwarm', ax=ax)
            ax.set_xlabel('R2 Score')
            st.pyplot(fig)

        st.markdown("---")
        
        brands = sorted(df_clean['Brand'].unique())
        selected_brand = st.selectbox("🚘 Select Brand", brands, key="pred_brand")

        filtered_models = sorted(df_clean[df_clean['Brand'] == selected_brand]['Model'].unique())
        selected_model = st.selectbox("🔧 Select Model", filtered_models, key="pred_model")

        filtered_rows = df_clean[(df_clean['Brand'] == selected_brand) & 
                                (df_clean['Model'] == selected_model)]

        if len(filtered_rows) > 0:
            st.markdown("### 🖼️ Car Gallery")
            
            for i in range(0, min(len(filtered_rows), 6), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(filtered_rows):
                        brand = filtered_rows.iloc[idx]['Brand']
                        model = filtered_rows.iloc[idx]['Model']
                        year = filtered_rows.iloc[idx]['Year']
                        
                        search_query = f"{brand}+{model}+{year}+car".replace(' ', '+')
                        img_url = f"https://tse1.mm.bing.net/th?q={search_query}&w=600&h=400&c=7&rs=1&p=0&dpr=1&pid=1.7&mkt=en-IN&adlt=moderate"
                        
                        try:
                            col.image(img_url, use_container_width=True, caption=f"{brand} {model} ({year})")
                        except:
                            col.info("Image not available")

            st.markdown("---")
            
            filtered_row = filtered_rows.iloc[0]
            
            st.markdown("### 🧩 Car Details")
            
            col1, col2, col3 = st.columns(3)
            inputs = {}
            
            feature_idx = 0
            for col in feature_columns:
                if col in filtered_row.index:
                    with [col1, col2, col3][feature_idx % 3]:
                        if df_clean[col].dtype == 'object':
                            options = sorted(df_clean[col].unique())
                            default = filtered_row[col]
                            inputs[col] = st.selectbox(f"{col}", options, index=options.index(default), key=f"pred_{col}")
                        else:
                            min_val = int(df_clean[col].min())
                            max_val = int(df_clean[col].max())
                            default_val = int(filtered_row[col])
                            inputs[col] = st.slider(f"{col}", min_val, max_val, default_val, key=f"pred_{col}")
                    feature_idx += 1

            col1, col2 = st.columns([3, 1])
            with col1:
                predict_btn = st.button("🔍 Predict Price", type="primary", use_container_width=True)
            with col2:
                add_wishlist = st.button("❤️ Add to Wishlist", use_container_width=True)
            
            if add_wishlist:
                wishlist_item = f"{selected_brand} {selected_model}"
                if wishlist_item not in st.session_state.wishlist:
                    st.session_state.wishlist.append(wishlist_item)
                    st.success("Added to wishlist!")
                else:
                    st.info("Already in wishlist!")
            
            if predict_btn:
                input_df = pd.DataFrame([inputs])
                for col in encoders:
                    if col in input_df:
                        input_df[col] = encoders[col].transform(input_df[col].astype(str))
                input_scaled = scaler.transform(input_df)
                predicted_price = best_model.predict(input_scaled)[0]

                st.markdown("---")
                st.subheader("📊 Price Estimation")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Minimum Price", f"₹{predicted_price*0.9:,.0f}", delta="-10%")
                with col2:
                    st.metric("Fair Market Price", f"₹{predicted_price:,.0f}", delta="Recommended")
                with col3:
                    st.metric("Maximum Price", f"₹{predicted_price*1.1:,.0f}", delta="+10%")
                
                st.balloons()
                
                # Save prediction
                st.session_state.predictions.append({
                    'Brand': selected_brand,
                    'Model': selected_model,
                    'Predicted_Price': f"₹{predicted_price:,.0f}",
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Save search history
                st.session_state.search_history.append({
                    'Search': f"{selected_brand} {selected_model}",
                    'Price': f"₹{predicted_price:,.0f}",
                    'Time': datetime.now().strftime("%H:%M:%S")
                })

    # ============================================
    # COMPARE CARS PAGE
    # ============================================
    elif page == "📊 Compare Cars":
        st.subheader("📊 Compare Multiple Cars Side-by-Side")
        
        num_cars = st.slider("Number of cars to compare", 2, 3, 2)
        
        comparison_data = []
        cols = st.columns(num_cars)
        
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"### Car {i+1}")
                brands = sorted(df_clean['Brand'].unique())
                brand = st.selectbox(f"Brand", brands, key=f"comp_brand_{i}")
                
                models = sorted(df_clean[df_clean['Brand'] == brand]['Model'].unique())
                model = st.selectbox(f"Model", models, key=f"comp_model_{i}")
                
                car_data = df_clean[(df_clean['Brand'] == brand) & (df_clean['Model'] == model)].iloc[0]
                
                search_query = f"{brand}+{model}+car".replace(' ', '+')
                img_url = f"https://tse1.mm.bing.net/th?q={search_query}&w=400&h=300&c=7&rs=1&p=0&dpr=1&pid=1.7&mkt=en-IN&adlt=moderate"
                try:
                    st.image(img_url, use_container_width=True)
                except:
                    st.info("Image not available")
                
                comparison_data.append({
                    'Brand': brand,
                    'Model': model,
                    'Price': car_data['Market_Price(INR)'],
                    'Year': car_data['Year'],
                    'Fuel_Type': car_data.get('Fuel_Type', 'N/A'),
                    'Transmission': car_data.get('Transmission', 'N/A'),
                    'Mileage': car_data.get('Mileage(km)', 'N/A'),
                    'Power_HP': car_data.get('Power_HP', 'N/A'),
                    'Engine_cc': car_data.get('Engine_cc', 'N/A')
                })
        
        if st.button("🔄 Compare Now", type="primary"):
            st.markdown("---")
            st.subheader("📋 Detailed Comparison")
            
            comparison_df = pd.DataFrame(comparison_data).T
            comparison_df.columns = [f"Car {i+1}" for i in range(num_cars)]
            st.dataframe(comparison_df, use_container_width=True)
            
            # Price comparison
            st.markdown("### 💰 Price Comparison")
            fig, ax = plt.subplots(figsize=(10, 5))
            car_names = [f"{d['Brand']} {d['Model']}" for d in comparison_data]
            prices = [d['Price'] for d in comparison_data]
            sns.barplot(x=car_names, y=prices, palette='Set2', ax=ax)
            ax.set_ylabel('Price (INR)')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            # Best value
            best_idx = prices.index(min(prices))
            worst_idx = prices.index(max(prices))
            st.success(f"💰 Best Value: **{comparison_data[best_idx]['Brand']} {comparison_data[best_idx]['Model']}** at ₹{comparison_data[best_idx]['Price']:,.0f}")
            st.info(f"💎 Premium Option: **{comparison_data[worst_idx]['Brand']} {comparison_data[worst_idx]['Model']}** at ₹{comparison_data[worst_idx]['Price']:,.0f}")

    # ============================================
    # EMI CALCULATOR PAGE
    # ============================================
    elif page == "🧮 EMI Calculator":
        st.subheader("🧮 Loan EMI Calculator")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Enter Loan Details")
            
            car_price = st.number_input("Car Price (₹)", min_value=100000, max_value=50000000, value=1000000, step=50000)
            down_payment = st.slider("Down Payment (%)", 0, 50, 20)
            interest_rate = st.slider("Annual Interest Rate (%)", 5.0, 20.0, 9.5, step=0.5)
            tenure_years = st.slider("Loan Tenure (Years)", 1, 7, 5)
            
            principal = car_price - (car_price * down_payment / 100)
            rate_monthly = interest_rate / (12 * 100)
            tenure_months = tenure_years * 12
            
            if rate_monthly > 0:
                emi = principal * rate_monthly * ((1 + rate_monthly)**tenure_months) / (((1 + rate_monthly)**tenure_months) - 1)
            else:
                emi = principal / tenure_months
            
            total_amount = emi * tenure_months
            total_interest = total_amount - principal
        
        with col2:
            st.markdown("### EMI Breakdown")
            
            st.metric("Monthly EMI", f"₹{emi:,.0f}")
            st.metric("Total Payable", f"₹{total_amount:,.0f}")
            st.metric("Total Interest", f"₹{total_interest:,.0f}")
            st.metric("Down Payment", f"₹{car_price * down_payment / 100:,.0f}")
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie([principal, total_interest], labels=['Principal', 'Interest'], 
                   autopct='%1.1f%%', startangle=90, colors=['#4A90E2', '#E24A4A'])
            ax.set_title('Loan Breakdown')
            st.pyplot(fig)
        
        st.markdown("---")
        
        st.subheader("📅 Payment Schedule (First 12 Months)")
        
        schedule = []
        balance = principal
        
        for month in range(1, min(13, tenure_months + 1)):
            interest_payment = balance * rate_monthly
            principal_payment = emi - interest_payment
            balance -= principal_payment
            
            schedule.append({
                'Month': month,
                'EMI': f"₹{emi:,.0f}",
                'Principal': f"₹{principal_payment:,.0f}",
                'Interest': f"₹{interest_payment:,.0f}",
                'Balance': f"₹{balance:,.0f}"
            })
        
        st.dataframe(pd.DataFrame(schedule), use_container_width=True, hide_index=True)
        
        # Bank Comparison
        st.markdown("---")
        st.subheader("🏦 Bank Loan Comparison")
        
        banks = {
            'HDFC Bank': 9.5,
            'SBI': 9.0,
            'ICICI Bank': 9.8,
            'Axis Bank': 9.7,
            'Kotak Mahindra': 9.6
        }
        
        bank_comparison = []
        for bank, rate in banks.items():
            rate_m = rate / (12 * 100)
            emi_bank = principal * rate_m * ((1 + rate_m)**tenure_months) / (((1 + rate_m)**tenure_months) - 1)
            total_bank = emi_bank * tenure_months
            bank_comparison.append({
                'Bank': bank,
                'Interest Rate': f"{rate}%",
                'Monthly EMI': f"₹{emi_bank:,.0f}",
                'Total Amount': f"₹{total_bank:,.0f}"
            })
        
        st.dataframe(pd.DataFrame(bank_comparison), use_container_width=True, hide_index=True)

    # ============================================
    # DEPRECIATION ANALYZER PAGE
    # ============================================
    elif page == "📉 Depreciation Analyzer":
        st.subheader("📉 Car Depreciation Analyzer")
        
        brands = sorted(df_clean['Brand'].unique())
        selected_brand = st.selectbox("Select Brand", brands, key="dep_brand")
        
        filtered_models = sorted(df_clean[df_clean['Brand'] == selected_brand]['Model'].unique())
        selected_model = st.selectbox("Select Model", filtered_models, key="dep_model")
        
        car_data = df_clean[(df_clean['Brand'] == selected_brand) & 
                           (df_clean['Model'] == selected_model)].iloc[0]
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Current Car Details")
            st.write(f"**Brand:** {selected_brand}")
            st.write(f"**Model:** {selected_model}")
            st.write(f"**Year:** {car_data['Year']}")
            st.write(f"**Current Price:** ₹{car_data['Market_Price(INR)']:,.0f}")
            
            current_value, future_values = calculate_depreciation(car_data['Market_Price(INR)'], car_data['Year'])
            
            st.markdown("### Depreciation Analysis")
            st.metric("Current Estimated Value", f"₹{current_value:,.0f}")
            st.metric("Total Depreciation", f"₹{car_data['Market_Price(INR)'] - current_value:,.0f}")
        
        with col2:
            st.markdown("### 5-Year Future Projection")
            
            future_df = pd.DataFrame(future_values)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(future_df['Year'], future_df['Value'], marker='o', linewidth=2, markersize=8, color='#E24A4A')
            ax.set_xlabel('Year')
            ax.set_ylabel('Estimated Value (INR)')
            ax.set_title('Depreciation Trend')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("📊 Future Value Breakdown")
        
        future_display = future_df.copy()
        future_display['Value'] = future_display['Value'].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(future_display, use_container_width=True, hide_index=True)
        
        st.info("💡 Cars typically depreciate 15% per year. Luxury cars may depreciate faster.")

    # ============================================
    # FUEL COST CALCULATOR PAGE
    # ============================================
    elif page == "⛽ Fuel Cost Calculator":
        st.subheader("⛽ Fuel Cost Calculator")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Enter Details")
            
            fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric', 'Hybrid'])
            mileage = st.number_input("Mileage (km per liter/kWh)", min_value=5.0, max_value=50.0, value=15.0, step=0.5)
            yearly_km = st.number_input("Yearly Kilometers", min_value=1000, max_value=50000, value=15000, step=1000)
            years = st.slider("Calculate for (years)", 1, 10, 5)
            
            yearly_cost = calculate_fuel_cost(mileage, fuel_type, yearly_km)
            total_cost = yearly_cost * years
        
        with col2:
            st.markdown("### Cost Breakdown")
            
            st.metric("Yearly Fuel Cost", f"₹{yearly_cost:,.0f}")
            st.metric("Monthly Fuel Cost", f"₹{yearly_cost/12:,.0f}")
            st.metric(f"Total Cost ({years} years)", f"₹{total_cost:,.0f}")
            
            fig, ax = plt.subplots(figsize=(6, 6))
            costs_by_fuel = []
            fuel_types = ['Petrol', 'Diesel', 'CNG', 'Electric', 'Hybrid']
            for ft in fuel_types:
                cost = calculate_fuel_cost(mileage, ft, yearly_km)
                costs_by_fuel.append(cost)
            
            ax.bar(fuel_types, costs_by_fuel, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            ax.set_ylabel('Yearly Cost (INR)')
            ax.set_title('Fuel Cost Comparison')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("💰 Savings Comparison")
        
        if fuel_type != 'Electric':
            electric_cost = calculate_fuel_cost(mileage * 1.5, 'Electric', yearly_km)
            savings = yearly_cost - electric_cost
            st.success(f"💡 Switching to Electric could save you ₹{savings:,.0f} per year!")

    # ============================================
    # TOTAL OWNERSHIP COST PAGE
    # ============================================
    elif page == "💎 Total Ownership Cost":
        st.subheader("💎 Total Cost of Ownership Calculator")
        
        brands = sorted(df_clean['Brand'].unique())
        selected_brand = st.selectbox("Select Brand", brands, key="tco_brand")
        
        filtered_models = sorted(df_clean[df_clean['Brand'] == selected_brand]['Model'].unique())
        selected_model = st.selectbox("Select Model", filtered_models, key="tco_model")
        
        car_data = df_clean[(df_clean['Brand'] == selected_brand) & 
                           (df_clean['Model'] == selected_model)].iloc[0]
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Input Parameters")
            
            car_price = car_data['Market_Price(INR)']
            st.write(f"**Car Price:** ₹{car_price:,.0f}")
            
            fuel_type = car_data.get('Fuel_Type', 'Petrol')
            mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=30.0, value=15.0)
            yearly_km = st.number_input("Yearly Distance (km)", min_value=5000, max_value=30000, value=15000)
            maintenance = st.number_input("Yearly Maintenance (₹)", min_value=10000, max_value=200000, value=50000)
            
            fuel_cost = calculate_fuel_cost(mileage, fuel_type, yearly_km)
            insurance = calculate_insurance(car_price, 2024 - car_data['Year'])
            
            yearly_total, total_5_years = calculate_total_ownership_cost(car_price, fuel_cost, insurance/car_price, maintenance)
        
        with col2:
            st.markdown("### Cost Breakdown")
            
            st.metric("Yearly Fuel Cost", f"₹{fuel_cost:,.0f}")
            st.metric("Yearly Insurance", f"₹{insurance:,.0f}")
            st.metric("Yearly Maintenance", f"₹{maintenance:,.0f}")
            st.metric("Total Yearly Cost", f"₹{yearly_total:,.0f}")
            st.metric("5-Year Ownership Cost", f"₹{total_5_years:,.0f}")
        
        st.markdown("---")
        
        # Pie chart
        st.subheader("📊 Cost Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        costs = [fuel_cost, insurance, maintenance]
        labels = ['Fuel', 'Insurance', 'Maintenance']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        ax.pie(costs, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Yearly Cost Distribution')
        st.pyplot(fig)
        
        st.info("💡 Total ownership cost includes depreciation, fuel, insurance, and maintenance over 5 years.")

    # ============================================
    # AI CHATBOT PAGE
    # ============================================
    elif page == "🤖 AI Chatbot":
        st.subheader("🤖 AI Car Assistant")
        
        st.markdown("### Ask me anything about cars!")
        st.write("Try: 'cars under 10 lakhs', 'best SUV', 'latest cars', 'cheapest sedan'")
        
        user_query = st.text_input("Your Question:", placeholder="e.g., Show me cars under 15 lakhs")
        
        if st.button("🔍 Ask AI", type="primary"):
            if user_query:
                response, cars_df = simple_chatbot(user_query, df_clean)
                
                st.success(f"🤖 {response}")
                
                if not cars_df.empty:
                    st.dataframe(cars_df[['Brand', 'Model', 'Year', 'Market_Price(INR)']].head(10), use_container_width=True, hide_index=True)
            else:
                st.warning("Please enter a question!")
        
        st.markdown("---")
        st.subheader("💡 Popular Questions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Cars under 10 lakhs"):
                response, cars_df = simple_chatbot("under 1000000", df_clean)
                st.success(response)
                if not cars_df.empty:
                    st.dataframe(cars_df[['Brand', 'Model', 'Market_Price(INR)']].head(5), use_container_width=True, hide_index=True)
            
            if st.button("Best SUVs"):
                response, cars_df = simple_chatbot("suv", df_clean)
                st.success(response)
                if not cars_df.empty:
                    st.dataframe(cars_df[['Brand', 'Model', 'Market_Price(INR)']].head(5), use_container_width=True, hide_index=True)
        
        with col2:
            if st.button("Latest Cars"):
                response, cars_df = simple_chatbot("latest", df_clean)
                st.success(response)
                if not cars_df.empty:
                    st.dataframe(cars_df[['Brand', 'Model', 'Year', 'Market_Price(INR)']].head(5), use_container_width=True, hide_index=True)
            
            if st.button("Cheapest Cars"):
                response, cars_df = simple_chatbot("cheap", df_clean)
                st.success(response)
                if not cars_df.empty:
                    st.dataframe(cars_df[['Brand', 'Model', 'Market_Price(INR)']].head(5), use_container_width=True, hide_index=True)

    # ============================================
    # REVIEWS & RATINGS PAGE
    # ============================================
    elif page == "⭐ Reviews & Ratings":
        st.subheader("⭐ Reviews & Ratings")
        
        if not st.session_state.logged_in:
            st.warning("⚠️ Please login to add reviews!")
        else:
            st.markdown("### Add Your Review")
            
            brands = sorted(df_clean['Brand'].unique())
            review_brand = st.selectbox("Select Brand", brands, key="review_brand")
            
            filtered_models = sorted(df_clean[df_clean['Brand'] == review_brand]['Model'].unique())
            review_model = st.selectbox("Select Model", filtered_models, key="review_model")
            
            rating = st.slider("Rating", 1, 5, 4)
            review_text = st.text_area("Your Review", placeholder="Share your experience...")
            
            if st.button("Submit Review", type="primary"):
                car_key = f"{review_brand} {review_model}"
                if car_key not in st.session_state.reviews:
                    st.session_state.reviews[car_key] = []
                
                st.session_state.reviews[car_key].append({
                    'User': st.session_state.username,
                    'Rating': rating,
                    'Review': review_text,
                    'Date': datetime.now().strftime("%Y-%m-%d")
                })
                st.success("Review submitted!")
        
        st.markdown("---")
        st.subheader("📝 All Reviews")
        
        if st.session_state.reviews:
            for car, reviews in st.session_state.reviews.items():
                with st.expander(f"{car} ({len(reviews)} reviews)"):
                    avg_rating = sum([r['Rating'] for r in reviews]) / len(reviews)
                    st.write(f"⭐ Average Rating: {avg_rating:.1f}/5")
                    
                    for review in reviews:
                        st.markdown(f"**{review['User']}** - {'⭐' * review['Rating']} ({review['Date']})")
                        st.write(review['Review'])
                        st.markdown("---")
        else:
            st.info("No reviews yet. Be the first to review!")

    # ============================================
    # WISHLIST PAGE
    # ============================================
    elif page == "❤️ Wishlist":
        st.subheader("❤️ My Wishlist")
        
        if not st.session_state.logged_in:
            st.warning("⚠️ Please login to access wishlist!")
        else:
            if st.session_state.wishlist:
                st.write(f"You have {len(st.session_state.wishlist)} cars in your wishlist")
                
                for idx, car in enumerate(st.session_state.wishlist):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"🚗 {car}")
                    with col2:
                        if st.button("Remove", key=f"remove_{idx}"):
                            st.session_state.wishlist.pop(idx)
                            st.rerun()
                
                if st.button("Clear All", type="secondary"):
                    st.session_state.wishlist = []
                    st.success("Wishlist cleared!")
                    st.rerun()
            else:
                st.info("Your wishlist is empty. Add cars from the prediction page!")

    # ============================================
    # PRICE ALERTS PAGE
    # ============================================
    elif page == "🔔 Price Alerts":
        st.subheader("🔔 Price Alerts")
        
        if not st.session_state.logged_in:
            st.warning("⚠️ Please login to set price alerts!")
        else:
            st.markdown("### Set Price Alert")
            
            brands = sorted(df_clean['Brand'].unique())
            alert_brand = st.selectbox("Brand", brands, key="alert_brand")
            
            filtered_models = sorted(df_clean[df_clean['Brand'] == alert_brand]['Model'].unique())
            alert_model = st.selectbox("Model", filtered_models, key="alert_model")
            
            target_price = st.number_input("Target Price (₹)", min_value=100000, max_value=50000000, value=1000000, step=50000)
            
            if st.button("Set Alert", type="primary"):
                st.session_state.price_alerts.append({
                    'Car': f"{alert_brand} {alert_model}",
                    'Target': target_price,
                    'Date': datetime.now().strftime("%Y-%m-%d")
                })
                st.success("Alert set! You'll be notified when price drops.")
            
            st.markdown("---")
            st.subheader("📋 Active Alerts")
            
            if st.session_state.price_alerts:
                for idx, alert in enumerate(st.session_state.price_alerts):
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(f"🚗 {alert['Car']}")
                    with col2:
                        st.write(f"Target: ₹{alert['Target']:,.0f}")
                    with col3:
                        if st.button("Delete", key=f"del_alert_{idx}"):
                            st.session_state.price_alerts.pop(idx)
                            st.rerun()
            else:
                st.info("No active alerts")

    # ============================================
    # MARKET INSIGHTS PAGE
    # ============================================
    elif page == "📈 Market Insights":
        st.subheader("📈 Market Insights & Analytics")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Distribution", "⛽ Fuel Analysis", "🏙️ City-wise", "📅 Year Trends"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(df_clean['Market_Price(INR)'], kde=True, bins=50, ax=ax, color='skyblue')
                ax.set_title('Price Distribution')
                ax.set_xlabel('Price (INR)')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(y=df_clean['Market_Price(INR)'], ax=ax, color='lightgreen')
                ax.set_title('Price Range Analysis')
                st.pyplot(fig)
        
        with tab2:
            if 'Fuel_Type' in df_clean.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(data=df_clean, x='Fuel_Type', y='Market_Price(INR)', ax=ax, palette='Set3')
                    ax.set_title('Price by Fuel Type')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                with col2:
                    fuel_counts = df_clean['Fuel_Type'].value_counts()
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(fuel_counts.values, labels=fuel_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.set_title('Fuel Type Distribution')
                    st.pyplot(fig)
        
        with tab3:
            if 'Registration_City' in df_clean.columns:
                city_avg = df_clean.groupby('Registration_City')['Market_Price(INR)'].mean().sort_values(ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=city_avg.values, y=city_avg.index, palette='rocket', ax=ax)
                ax.set_xlabel('Average Price (INR)')
                ax.set_title('Average Price by City (Top 10)')
                st.pyplot(fig)
        
        with tab4:
            year_avg = df_clean.groupby('Year')['Market_Price(INR)'].mean().sort_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(year_avg.index, year_avg.values, marker='o', linewidth=2, markersize=8, color='#4A90E2')
            ax.set_xlabel('Year')
            ax.set_ylabel('Average Price (INR)')
            ax.set_title('Average Price Trend by Year')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # ============================================
    # DOWNLOAD REPORT PAGE
    # ============================================
    elif page == "📥 Download Report":
        st.subheader("📥 Download Reports & Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Available Reports")
            
            csv_full = df_clean.to_csv(index=False).encode('utf-8')
            st.download_button("📄 Download Full Dataset", csv_full, "full_dataset.csv", "text/csv", key='dl1')
            
            csv_model = result_df.to_csv().encode('utf-8')
            st.download_button("🤖 Model Performance", csv_model, "model_performance.csv", "text/csv", key='dl2')
            
            summary_stats = df_clean['Market_Price(INR)'].describe().to_frame()
            csv_summary = summary_stats.to_csv().encode('utf-8')
            st.download_button("📈 Price Summary", csv_summary, "price_summary.csv", "text/csv", key='dl3')
        
        with col2:
            st.markdown("### 📋 Your Activity")
            
            if st.session_state.predictions:
                pred_df = pd.DataFrame(st.session_state.predictions)
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                
                csv_pred = pred_df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Predictions", csv_pred, "my_predictions.csv", "text/csv", key='dl4')
            else:
                st.info("No predictions yet!")

else:
    st.info("📥 Please upload your dataset to start!")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **💰 Price Prediction**
        - AI models
        - Real images
        - Instant results
        """)
    
    with col2:
        st.markdown("""
        **🧮 Calculators**
        - EMI Calculator
        - Fuel Cost
        - Depreciation
        """)
    
    with col3:
        st.markdown("""
        **🤖 AI Features**
        - Chatbot
        - Recommendations
        - Smart search
        """)
    
    with col4:
        st.markdown("""
        **⭐ Social**
        - Reviews
        - Wishlist
        - Price alerts
        """)
