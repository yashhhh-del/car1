# ======================================================
# SMART PRICING SYSTEM FOR USED CARS - ULTIMATE PRO VERSION
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
import base64
from io import BytesIO

# Page config
st.set_page_config(page_title="Smart Car Pricing ULTIMATE PRO", layout="wide", initial_sidebar_state="expanded")

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
if 'inspection_checklist' not in st.session_state:
    st.session_state.inspection_checklist = {}
if 'maintenance_log' not in st.session_state:
    st.session_state.maintenance_log = []

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
        .feature-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
        }
        .whatsapp-btn {
            background-color: #25D366;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            display: inline-block;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚗 Smart Car Pricing System ULTIMATE PRO</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Predictions | Advanced Analytics | Tax Calculator | Inspection Checklist & More!")

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

def calculate_resale_value(price, year, future_years=3):
    current_year = 2024
    age = current_year - year
    current_value = price * ((1 - 0.15) ** age)
    
    resale_values = []
    for i in range(1, future_years + 1):
        future_value = current_value * ((1 - 0.12) ** i)
        resale_values.append({
            'Years from now': i,
            'Year': current_year + i,
            'Estimated Value': future_value,
            'Depreciation': current_value - future_value
        })
    return resale_values

def best_time_to_sell(price, year):
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    
    demand_multiplier = {
        'January': 0.95, 'February': 0.97, 'March': 1.05, 'April': 1.08,
        'May': 0.92, 'June': 0.90, 'July': 0.93, 'August': 0.96,
        'September': 1.02, 'October': 1.10, 'November': 1.12, 'December': 1.08
    }
    
    seasonal_prices = []
    for month in months:
        seasonal_price = price * demand_multiplier[month]
        seasonal_prices.append({'Month': month, 'Expected Price': seasonal_price})
    
    best_month = max(seasonal_prices, key=lambda x: x['Expected Price'])
    worst_month = min(seasonal_prices, key=lambda x: x['Expected Price'])
    
    return seasonal_prices, best_month, worst_month

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

def calculate_road_tax(car_price, state='Maharashtra'):
    tax_rates = {
        'Maharashtra': 0.13,
        'Delhi': 0.10,
        'Karnataka': 0.13,
        'Tamil Nadu': 0.20,
        'Gujarat': 0.13,
        'Uttar Pradesh': 0.08
    }
    rate = tax_rates.get(state, 0.10)
    return car_price * rate

def calculate_gst(car_price):
    if car_price <= 1000000:
        return car_price * 0.28
    else:
        return car_price * 0.28 + (car_price - 1000000) * 0.20

def calculate_on_road_price(ex_showroom_price, state='Maharashtra'):
    road_tax = calculate_road_tax(ex_showroom_price, state)
    insurance = ex_showroom_price * 0.03
    registration = 5000
    other_charges = 10000
    
    on_road_price = ex_showroom_price + road_tax + insurance + registration + other_charges
    
    return {
        'Ex-Showroom': ex_showroom_price,
        'Road Tax': road_tax,
        'Insurance': insurance,
        'Registration': registration,
        'Other Charges': other_charges,
        'On-Road Price': on_road_price
    }

def loan_vs_lease_comparison(car_price, down_payment_percent=20, loan_years=5, lease_years=3):
    # Loan calculation
    loan_amount = car_price * (1 - down_payment_percent/100)
    loan_rate = 0.095
    loan_months = loan_years * 12
    loan_emi = loan_amount * (loan_rate/12) * ((1 + loan_rate/12)**loan_months) / (((1 + loan_rate/12)**loan_months) - 1)
    total_loan_cost = loan_emi * loan_months + (car_price * down_payment_percent/100)
    
    # Lease calculation
    lease_rate = 0.08
    lease_months = lease_years * 12
    residual_value = car_price * 0.50
    lease_amount = car_price - residual_value
    lease_emi = lease_amount * (lease_rate/12) * ((1 + lease_rate/12)**lease_months) / (((1 + lease_rate/12)**lease_months) - 1)
    total_lease_cost = lease_emi * lease_months
    
    return {
        'Loan EMI': loan_emi,
        'Total Loan Cost': total_loan_cost,
        'Lease EMI': lease_emi,
        'Total Lease Cost': total_lease_cost,
        'Savings': total_loan_cost - total_lease_cost if total_loan_cost > total_lease_cost else 0
    }

def generate_whatsapp_link(car_details):
    brand = car_details.get('Brand', 'Car')
    model = car_details.get('Model', '')
    price = car_details.get('Price', 0)
    
    message = f"Check out this amazing car deal!\n\n🚗 {brand} {model}\n💰 Price: ₹{price:,.0f}\n\n📱 View more details on our platform!"
    encoded_message = message.replace('\n', '%0A').replace(' ', '%20')
    
    return f"https://wa.me/?text={encoded_message}"

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
    elif 'electric' in query or 'ev' in query:
        cars = df[df['Fuel_Type'].str.contains('Electric', case=False, na=False)].head(5) if 'Fuel_Type' in df.columns else pd.DataFrame()
        return "Electric vehicles:", cars
    else:
        return "Try asking: 'cars under 10 lakhs', 'best SUV', 'latest cars', 'electric cars'", pd.DataFrame()

def get_inspection_checklist():
    return {
        'Exterior': [
            'Body condition (dents, scratches)',
            'Paint quality and consistency',
            'Rust or corrosion',
            'Windshield condition',
            'Lights (headlights, taillights)',
            'Tires condition and tread depth',
            'Wheels and rims'
        ],
        'Interior': [
            'Seats condition',
            'Dashboard and controls',
            'Air conditioning',
            'Audio system',
            'Odometer reading',
            'Seat belts functionality',
            'Interior cleanliness'
        ],
        'Engine & Mechanical': [
            'Engine sound',
            'Oil level and condition',
            'Coolant level',
            'Battery condition',
            'Brake system',
            'Suspension',
            'Transmission performance'
        ],
        'Documents': [
            'Registration Certificate (RC)',
            'Insurance papers',
            'Pollution certificate',
            'Service records',
            'Owner manual',
            'Previous accident records',
            'NOC (if applicable)'
        ],
        'Test Drive': [
            'Acceleration',
            'Braking',
            'Steering response',
            'Gear shifting',
            'Noise levels',
            'Vibrations',
            'Overall driving comfort'
        ]
    }

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
        "💳 Loan vs Lease",
        "📉 Depreciation Analyzer",
        "💎 Resale Value Predictor",
        "📅 Best Time to Sell",
        "⛽ Fuel Cost Calculator",
        "💰 Total Ownership Cost",
        "🧾 Tax Calculator",
        "🔍 Car Inspection Checklist",
        "🔧 Maintenance Tracker",
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
        
        # Quick Search
        st.subheader("🔍 Advanced Search")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            search_brand = st.multiselect("Brand", ["All"] + sorted(df_clean['Brand'].unique().tolist()), key="home_brand")
        with col2:
            if 'Fuel_Type' in df_clean.columns:
                search_fuel = st.multiselect("Fuel Type", ["All"] + sorted(df_clean['Fuel_Type'].unique().tolist()), key="home_fuel")
            else:
                search_fuel = ["All"]
        with col3:
            if 'Transmission' in df_clean.columns:
                search_trans = st.multiselect("Transmission", ["All"] + sorted(df_clean['Transmission'].unique().tolist()), key="home_trans")
            else:
                search_trans = ["All"]
        with col4:
            year_range = st.slider("Year Range", int(df_clean['Year'].min()), int(df_clean['Year'].max()), 
                                  (int(df_clean['Year'].min()), int(df_clean['Year'].max())), key="home_year")
        
        price_range = st.slider("Price Range (Lakhs)", 0, int(df_clean['Market_Price(INR)'].max()/100000), (0, 50), key="home_price")
        
        filtered_data = df_clean.copy()
        if search_brand and "All" not in search_brand:
            filtered_data = filtered_data[filtered_data['Brand'].isin(search_brand)]
        if search_fuel and "All" not in search_fuel and 'Fuel_Type' in df_clean.columns:
            filtered_data = filtered_data[filtered_data['Fuel_Type'].isin(search_fuel)]
        if search_trans and "All" not in search_trans and 'Transmission' in df_clean.columns:
            filtered_data = filtered_data[filtered_data['Transmission'].isin(search_trans)]
        
        filtered_data = filtered_data[(filtered_data['Market_Price(INR)'] >= price_range[0]*100000) & 
                                     (filtered_data['Market_Price(INR)'] <= price_range[1]*100000)]
        filtered_data = filtered_data[(filtered_data['Year'] >= year_range[0]) & (filtered_data['Year'] <= year_range[1])]
        
        st.write(f"Found {len(filtered_data)} cars matching your criteria")
        st.dataframe(filtered_data.head(20), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            csv = filtered_data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "search_results.csv", "text/csv")
        with col2:
            if len(filtered_data) > 0:
                sample_car = filtered_data.iloc[0]
                whatsapp_link = generate_whatsapp_link({
                    'Brand': sample_car['Brand'],
                    'Model': sample_car['Model'],
                    'Price': sample_car['Market_Price(INR)']
                })
                st.markdown(f'<a href="{whatsapp_link}" target="_blank" class="whatsapp-btn">📱 Share on WhatsApp</a>', unsafe_allow_html=True)

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

            col1, col2, col3 = st.columns(3)
            with col1:
                predict_btn = st.button("🔍 Predict Price", type="primary", use_container_width=True)
            with col2:
                add_wishlist = st.button("❤️ Add to Wishlist", use_container_width=True)
            with col3:
                share_whatsapp = st.button("📱 Share", use_container_width=True)
            
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
                
                # WhatsApp share link
                whatsapp_link = generate_whatsapp_link({
                    'Brand': selected_brand,
                    'Model': selected_model,
                    'Price': predicted_price
                })
                st.markdown(f'<a href="{whatsapp_link}" target="_blank" class="whatsapp-btn">📱 Share Price on WhatsApp</a>', unsafe_allow_html=True)

    # ============================================
    # COMPARE CARS PAGE
    # ============================================
    elif page == "📊 Compare Cars":
        st.subheader("📊 Compare Multiple Cars")
        
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
            
            st.markdown("### 💰 Price Comparison")
            fig, ax = plt.subplots(figsize=(10, 5))
            car_names = [f"{d['Brand']} {d['Model']}" for d in comparison_data]
            prices = [d['Price'] for d in comparison_data]
            sns.barplot(x=car_names, y=prices, palette='Set2', ax=ax)
            ax.set_ylabel('Price (INR)')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            best_idx = prices.index(min(prices))
            worst_idx = prices.index(max(prices))
            st.success(f"💰 Best Value: **{comparison_data[best_idx]['Brand']} {comparison_data[best_idx]['Model']}** at ₹{comparison_data[best_idx]['Price']:,.0f}")
            st.info(f"💎 Premium Option: **{comparison_data[worst_idx]['Brand']} {comparison_data[worst_idx]['Model']}** at ₹{comparison_data[worst_idx]['Price']:,.0f}")

    # ============================================
    # EMI CALCULATOR
    # ============================================
    elif page == "🧮 EMI Calculator":
        st.subheader("🧮 Loan EMI Calculator")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Enter Details")
            car_price = st.number_input("Car Price (₹)", min_value=100000, max_value=50000000, value=1000000, step=50000)
            down_payment = st.slider("Down Payment (%)", 0, 50, 20)
            interest_rate = st.slider("Interest Rate (%)", 5.0, 20.0, 9.5, step=0.5)
            tenure_years = st.slider("Tenure (Years)", 1, 7, 5)
            
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
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie([principal, total_interest], labels=['Principal', 'Interest'], 
                   autopct='%1.1f%%', startangle=90, colors=['#4A90E2', '#E24A4A'])
            st.pyplot(fig)

    # ============================================
    # LOAN VS LEASE
    # ============================================
    elif page == "💳 Loan vs Lease":
        st.subheader("💳 Loan vs Lease Comparison")
        
        car_price = st.number_input("Car Price (₹)", min_value=500000, max_value=50000000, value=2000000, step=100000)
        
        comparison = loan_vs_lease_comparison(car_price)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏦 Loan Option")
            st.metric("Monthly EMI", f"₹{comparison['Loan EMI']:,.0f}")
            st.metric("Total Cost", f"₹{comparison['Total Loan Cost']:,.0f}")
            st.info("✅ You own the car at the end")
        
        with col2:
            st.markdown("### 📋 Lease Option")
            st.metric("Monthly Payment", f"₹{comparison['Lease EMI']:,.0f}")
            st.metric("Total Cost", f"₹{comparison['Total Lease Cost']:,.0f}")
            st.info("⚠️ Return car after lease period")
        
        if comparison['Savings'] > 0:
            st.success(f"💰 Leasing saves you ₹{comparison['Savings']:,.0f}!")
        else:
            st.success(f"💰 Buying saves you ₹{abs(comparison['Savings']):,.0f}!")

    # ============================================
    # DEPRECIATION ANALYZER
    # ============================================
    elif page == "📉 Depreciation Analyzer":
        st.subheader("📉 Car Depreciation Analysis")
        
        brands = sorted(df_clean['Brand'].unique())
        selected_brand = st.selectbox("Brand", brands, key="dep_brand")
        
        filtered_models = sorted(df_clean[df_clean['Brand'] == selected_brand]['Model'].unique())
        selected_model = st.selectbox("Model", filtered_models, key="dep_model")
        
        car_data = df_clean[(df_clean['Brand'] == selected_brand) & (df_clean['Model'] == selected_model)].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Current Details")
            st.write(f"**Price:** ₹{car_data['Market_Price(INR)']:,.0f}")
            st.write(f"**Year:** {car_data['Year']}")
            
            current_value, future_values = calculate_depreciation(car_data['Market_Price(INR)'], car_data['Year'])
            st.metric("Current Value", f"₹{current_value:,.0f}")
        
        with col2:
            st.markdown("### 5-Year Projection")
            future_df = pd.DataFrame(future_values)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(future_df['Year'], future_df['Value'], marker='o', color='#E24A4A')
            ax.set_xlabel('Year')
            ax.set_ylabel('Value (INR)')
            st.pyplot(fig)

    # ============================================
    # RESALE VALUE PREDICTOR
    # ============================================
    elif page == "💎 Resale Value Predictor":
        st.subheader("💎 Resale Value Predictor")
        
        brands = sorted(df_clean['Brand'].unique())
        selected_brand = st.selectbox("Brand", brands, key="resale_brand")
        
        filtered_models = sorted(df_clean[df_clean['Brand'] == selected_brand]['Model'].unique())
        selected_model = st.selectbox("Model", filtered_models, key="resale_model")
        
        car_data = df_clean[(df_clean['Brand'] == selected_brand) & (df_clean['Model'] == selected_model)].iloc[0]
        
        years_ahead = st.slider("Predict for (years)", 1, 5, 3)
        
        resale_values = calculate_resale_value(car_data['Market_Price(INR)'], car_data['Year'], years_ahead)
        
        resale_df = pd.DataFrame(resale_values)
        resale_df['Estimated Value'] = resale_df['Estimated Value'].apply(lambda x: f"₹{x:,.0f}")
        resale_df['Depreciation'] = resale_df['Depreciation'].apply(lambda x: f"₹{x:,.0f}")
        
        st.dataframe(resale_df, use_container_width=True, hide_index=True)

    # ============================================
    # BEST TIME TO SELL
    # ============================================
    elif page == "📅 Best Time to Sell":
        st.subheader("📅 Best Time to Sell Analysis")
        
        brands = sorted(df_clean['Brand'].unique())
        selected_brand = st.selectbox("Brand", brands, key="time_brand")
        
        filtered_models = sorted(df_clean[df_clean['Brand'] == selected_brand]['Model'].unique())
        selected_model = st.selectbox("Model", filtered_models, key="time_model")
        
        car_data = df_clean[(df_clean['Brand'] == selected_brand) & (df_clean['Model'] == selected_model)].iloc[0]
        
        seasonal_prices, best_month, worst_month = best_time_to_sell(car_data['Market_Price(INR)'], car_data['Year'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"🌟 Best Month: **{best_month['Month']}**")
            st.metric("Expected Price", f"₹{best_month['Expected Price']:,.0f}")
        
        with col2:
            st.error(f"⚠️ Worst Month: **{worst_month['Month']}**")
            st.metric("Expected Price", f"₹{worst_month['Expected Price']:,.0f}")
        
        st.markdown("---")
        seasonal_df = pd.DataFrame(seasonal_prices)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(seasonal_df['Month'], seasonal_df['Expected Price'], marker='o', linewidth=2)
        ax.set_xlabel('Month')
        ax.set_ylabel('Expected Price (INR)')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ============================================
    # FUEL COST CALCULATOR
    # ============================================
    elif page == "⛽ Fuel Cost Calculator":
        st.subheader("⛽ Fuel Cost Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric', 'Hybrid'])
            mileage = st.number_input("Mileage (km/l)", 5.0, 50.0, 15.0)
            yearly_km = st.number_input("Yearly KM", 1000, 50000, 15000)
            years = st.slider("Calculate for (years)", 1, 10, 5)
            
            yearly_cost = calculate_fuel_cost(mileage, fuel_type, yearly_km)
            total_cost = yearly_cost * years
        
        with col2:
            st.metric("Yearly Cost", f"₹{yearly_cost:,.0f}")
            st.metric("Monthly Cost", f"₹{yearly_cost/12:,.0f}")
            st.metric(f"Total ({years} years)", f"₹{total_cost:,.0f}")

    # ============================================
    # TOTAL OWNERSHIP COST
    # ============================================
    elif page == "💰 Total Ownership Cost":
        st.subheader("💰 Total Ownership Cost")
        
        brands = sorted(df_clean['Brand'].unique())
        selected_brand = st.selectbox("Brand", brands, key="tco_brand")
        
        filtered_models = sorted(df_clean[df_clean['Brand'] == selected_brand]['Model'].unique())
        selected_model = st.selectbox("Model", filtered_models, key="tco_model")
        
        car_data = df_clean[(df_clean['Brand'] == selected_brand) & (df_clean['Model'] == selected_model)].iloc[0]
        
        car_price = car_data['Market_Price(INR)']
        fuel_type = car_data.get('Fuel_Type', 'Petrol')
        
        mileage = st.number_input("Mileage (km/l)", 5.0, 30.0, 15.0)
        yearly_km = st.number_input("Yearly KM", 5000, 30000, 15000)
        maintenance = st.number_input("Yearly Maintenance (₹)", 10000, 200000, 50000)
        
        fuel_cost = calculate_fuel_cost(mileage, fuel_type, yearly_km)
        insurance = calculate_insurance(car_price, 2024 - car_data['Year'])
        yearly_total, total_5_years = calculate_total_ownership_cost(car_price, fuel_cost, insurance/car_price, maintenance)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Yearly Cost", f"₹{yearly_total:,.0f}")
            st.metric("5-Year Total", f"₹{total_5_years:,.0f}")
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 6))
            costs = [fuel_cost, insurance, maintenance]
            labels = ['Fuel', 'Insurance', 'Maintenance']
            ax.pie(costs, labels=labels, autopct='%1.1f%%', startangle=90)
            st.pyplot(fig)

    # ============================================
    # TAX CALCULATOR
    # ============================================
    elif page == "🧾 Tax Calculator":
        st.subheader("🧾 Tax & On-Road Price Calculator")
        
        ex_showroom = st.number_input("Ex-Showroom Price (₹)", 500000, 50000000, 2000000, 100000)
        state = st.selectbox("State", ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'Gujarat', 'Uttar Pradesh'])
        
        breakdown = calculate_on_road_price(ex_showroom, state)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Price Breakdown")
            for key, value in breakdown.items():
                if key != 'On-Road Price':
                    st.write(f"**{key}:** ₹{value:,.0f}")
        
        with col2:
            st.markdown("### Total On-Road Price")
            st.metric("Final Price", f"₹{breakdown['On-Road Price']:,.0f}")
            
            fig, ax = plt.subplots(figsize=(6, 6))
            components = ['Ex-Showroom', 'Road Tax', 'Insurance', 'Others']
            values = [breakdown['Ex-Showroom'], breakdown['Road Tax'], breakdown['Insurance'], 
                     breakdown['Registration'] + breakdown['Other Charges']]
            ax.pie(values, labels=components, autopct='%1.1f%%', startangle=90)
            st.pyplot(fig)

    # ============================================
    # CAR INSPECTION CHECKLIST
    # ============================================
    elif page == "🔍 Car Inspection Checklist":
        st.subheader("🔍 Pre-Purchase Car Inspection Checklist")
        
        if not st.session_state.logged_in:
            st.warning("⚠️ Please login to save inspection results!")
        
        car_id = st.text_input("Car Identification (e.g., Brand Model)", key="inspect_car")
        
        checklist = get_inspection_checklist()
        
        scores = {}
        
        for category, items in checklist.items():
            with st.expander(f"📋 {category}", expanded=True):
                category_score = 0
                for item in items:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(item)
                    with col2:
                        status = st.checkbox("✓", key=f"check_{category}_{item}")
                        if status:
                            category_score += 1
                
                scores[category] = (category_score / len(items)) * 100
                st.progress(category_score / len(items))
                st.write(f"Score: {category_score}/{len(items)}")
        
        if st.button("💾 Save Inspection Report", type="primary"):
            if st.session_state.logged_in and car_id:
                st.session_state.inspection_checklist[car_id] = {
                    'scores': scores,
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'user': st.session_state.username
                }
                st.success("Inspection report saved!")
        
        overall_score = sum(scores.values()) / len(scores)
        
        st.markdown("---")
        st.subheader("📊 Overall Inspection Score")
        st.metric("Total Score", f"{overall_score:.1f}%")
        
        if overall_score >= 80:
            st.success("✅ Excellent condition! Safe to buy.")
        elif overall_score >= 60:
            st.warning("⚠️ Good condition with minor issues.")
        else:
            st.error("❌ Multiple issues found. Proceed with caution!")

    # ============================================
    # MAINTENANCE TRACKER
    # ============================================
    elif page == "🔧 Maintenance Tracker":
        st.subheader("🔧 Car Maintenance Tracker")
        
        if not st.session_state.logged_in:
            st.warning("⚠️ Please login to track maintenance!")
        else:
            st.markdown("### Add Maintenance Record")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                car_name = st.text_input("Car Name")
            with col2:
                service_type = st.selectbox("Service Type", ['Oil Change', 'Tire Rotation', 'Brake Service', 'General Service', 'Other'])
            with col3:
                cost = st.number_input("Cost (₹)", 0, 100000, 5000)
            
            date = st.date_input("Service Date")
            notes = st.text_area("Notes")
            
            if st.button("Add Record"):
                st.session_state.maintenance_log.append({
                    'Car': car_name,
                    'Service': service_type,
                    'Cost': cost,
                    'Date': date.strftime("%Y-%m-%d"),
                    'Notes': notes
                })
                st.success("Maintenance record added!")
            
            if st.session_state.maintenance_log:
                st.markdown("---")
                st.subheader("📋 Maintenance History")
                
                log_df = pd.DataFrame(st.session_state.maintenance_log)
                st.dataframe(log_df, use_container_width=True, hide_index=True)
                
                total_spent = sum([r['Cost'] for r in st.session_state.maintenance_log])
                st.metric("Total Spent", f"₹{total_spent:,.0f}")

    # ============================================
    # AI CHATBOT
    # ============================================
    elif page == "🤖 AI Chatbot":
        st.subheader("🤖 AI Car Assistant")
        
        user_query = st.text_input("Ask me anything:", placeholder="e.g., cars under 10 lakhs")
        
        if st.button("🔍 Ask"):
            if user_query:
                response, cars_df = simple_chatbot(user_query, df_clean)
                st.success(f"🤖 {response}")
                if not cars_df.empty:
                    st.dataframe(cars_df[['Brand', 'Model', 'Year', 'Market_Price(INR)']].head(10), use_container_width=True, hide_index=True)

    # ============================================
    # REVIEWS & RATINGS
    # ============================================
    elif page == "⭐ Reviews & Ratings":
        st.subheader("⭐ Reviews & Ratings")
        
        if st.session_state.logged_in:
            brands = sorted(df_clean['Brand'].unique())
            review_brand = st.selectbox("Brand", brands, key="review_brand")
            filtered_models = sorted(df_clean[df_clean['Brand'] == review_brand]['Model'].unique())
            review_model = st.selectbox("Model", filtered_models, key="review_model")
            
            rating = st.slider("Rating", 1, 5, 4)
            review_text = st.text_area("Review")
            
            if st.button("Submit"):
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

    # ============================================
    # WISHLIST
    # ============================================
    elif page == "❤️ Wishlist":
        st.subheader("❤️ My Wishlist")
        
        if st.session_state.logged_in:
            if st.session_state.wishlist:
                for idx, car in enumerate(st.session_state.wishlist):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"🚗 {car}")
                    with col2:
                        if st.button("Remove", key=f"rm_{idx}"):
                            st.session_state.wishlist.pop(idx)
                            st.rerun()
            else:
                st.info("Wishlist empty!")

    # ============================================
    # PRICE ALERTS
    # ============================================
    elif page == "🔔 Price Alerts":
        st.subheader("🔔 Price Alerts")
        
        if st.session_state.logged_in:
            brands = sorted(df_clean['Brand'].unique())
            alert_brand = st.selectbox("Brand", brands, key="alert_brand")
            filtered_models = sorted(df_clean[df_clean['Brand'] == alert_brand]['Model'].unique())
            alert_model = st.selectbox("Model", filtered_models, key="alert_model")
            target_price = st.number_input("Target Price (₹)", 100000, 50000000, 1000000)
            
            if st.button("Set Alert"):
                st.session_state.price_alerts.append({
                    'Car': f"{alert_brand} {alert_model}",
                    'Target': target_price
                })
                st.success("Alert set!")

    # ============================================
    # MARKET INSIGHTS
    # ============================================
    elif page == "📈 Market Insights":
        st.subheader("📈 Market Insights")
        
        tab1, tab2 = st.tabs(["Price Distribution", "Fuel Analysis"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df_clean['Market_Price(INR)'], kde=True, bins=50, ax=ax)
            st.pyplot(fig)
        
        with tab2:
            if 'Fuel_Type' in df_clean.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=df_clean, x='Fuel_Type', y='Market_Price(INR)', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

    # ============================================
    # DOWNLOAD REPORT
    # ============================================
    elif page == "📥 Download Report":
        st.subheader("📥 Download Reports")
        
        csv_full = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Download Dataset", csv_full, "dataset.csv", "text/csv")
        
        if st.session_state.predictions:
            pred_df = pd.DataFrame(st.session_state.predictions)
            csv_pred = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Predictions", csv_pred, "predictions.csv", "text/csv")

else:
    st.info("📥 Please upload dataset to start!")
    st.markdown("### 🎯 Complete Feature List:")
    st.write("✅ 19 Advanced Features | AI Predictions | Tax Calculator | Inspection Checklist | Maintenance Tracker & More!")
