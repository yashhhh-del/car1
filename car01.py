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
            st.
