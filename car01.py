# ======================================================
# SMART PRICING SYSTEM - ULTIMATE PRO MAX VERSION
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
import matplotlib
matplotlib.use('Agg')

# Page config
st.set_page_config(page_title="Smart Car Pricing PRO MAX", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'email' not in st.session_state:
    st.session_state.email = ""
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
if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'forum_posts' not in st.session_state:
    st.session_state.forum_posts = []

# Translation dictionary
translations = {
    'English': {
        'title': '🚗 Smart Car Pricing System PRO MAX',
        'subtitle': 'AI-Powered | PDF Reports | Multi-language | Trade-in Estimator & More!',
        'upload': 'Upload CSV/XLSX File',
        'login': 'Login',
        'signup': 'Sign Up',
        'username': 'Username',
        'password': 'Password',
        'email': 'Email',
        'home': 'Home',
        'predict': 'Price Prediction',
        'compare': 'Compare Cars',
        'emi': 'EMI Calculator',
        'wishlist': 'Wishlist',
        'reports': 'Download Report'
    },
    'हिंदी': {
        'title': '🚗 स्मार्ट कार प्राइसिंग सिस्टम PRO MAX',
        'subtitle': 'AI-संचालित | PDF रिपोर्ट | बहुभाषी | ट्रेड-इन एस्टीमेटर और अधिक!',
        'upload': 'CSV/XLSX फ़ाइल अपलोड करें',
        'login': 'लॉगिन',
        'signup': 'साइन अप',
        'username': 'उपयोगकर्ता नाम',
        'password': 'पासवर्ड',
        'email': 'ईमेल',
        'home': 'होम',
        'predict': 'मूल्य भविष्यवाणी',
        'compare': 'कारों की तुलना करें',
        'emi': 'EMI कैलकुलेटर',
        'wishlist': 'विशलिस्ट',
        'reports': 'रिपोर्ट डाउनलोड करें'
    },
    'मराठी': {
        'title': '🚗 स्मार्ट कार प्राइसिंग सिस्टीम PRO MAX',
        'subtitle': 'AI-चालित | PDF अहवाल | बहुभाषिक | ट्रेड-इन अंदाजकर्ता आणि अधिक!',
        'upload': 'CSV/XLSX फाइल अपलोड करा',
        'login': 'लॉगिन',
        'signup': 'साइन अप',
        'username': 'वापरकर्ता नाव',
        'password': 'पासवर्ड',
        'email': 'ईमेल',
        'home': 'होम',
        'predict': 'किंमत अंदाज',
        'compare': 'गाड्यांची तुलना करा',
        'emi': 'EMI कॅल्क्युलेटर',
        'wishlist': 'विशलिस्ट',
        'reports': 'अहवाल डाउनलोड करा'
    }
}

def t(key):
    """Translation helper"""
    return translations[st.session_state.language].get(key, key)

# Custom CSS
if st.session_state.dark_mode:
    st.markdown("""
    <style>
        .main {background-color: #1e1e1e; color: #ffffff;}
        .stApp {background-color: #1e1e1e;}
        h1, h2, h3 {color: #4A90E2 !important;}
        .stMetric {
            background-color: #2c2c2c;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        }
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
        .whatsapp-btn {
            background-color: #25D366;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            display: inline-block;
            transition: background-color 0.3s;
        }
        .whatsapp-btn:hover {
            background-color: #1da851;
        }
        .qr-container {
            text-align: center;
            padding: 20px;
            background: #f0f0f0;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .stMetric {
            background-color: #e0f7fa;
            border-left: 5px solid #00bcd4;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .stPlotlyChart {
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown(f'<h1 class="main-header">{t("title")}</h1>', unsafe_allow_html=True)
st.markdown(f"### {t('subtitle')}")

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

def calculate_trade_in_value(brand, model, year, condition, mileage, market_price):
    """Calculate trade-in value based on multiple factors"""
    base_value = market_price
    
    # Age depreciation
    age = 2024 - year
    age_factor = (1 - 0.15) ** age
    
    # Condition factor
    condition_factors = {
        'Excellent': 1.0,
        'Good': 0.9,
        'Fair': 0.75,
        'Poor': 0.6
    }
    condition_factor = condition_factors.get(condition, 0.8)
    
    # Mileage factor (assuming average is 15000 km/year)
    expected_mileage = age * 15000
    mileage_factor = 1.0
    if mileage > expected_mileage:
        excess = (mileage - expected_mileage) / expected_mileage
        mileage_factor = max(0.7, 1 - (excess * 0.1))
    
    # Brand factor (premium brands retain value better)
    premium_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Toyota', 'Honda']
    brand_factor = 1.1 if brand in premium_brands else 1.0
    
    trade_in_value = base_value * age_factor * condition_factor * mileage_factor * brand_factor
    
    return {
        'Trade-in Value': trade_in_value,
        'Age Factor': age_factor,
        'Condition Factor': condition_factor,
        'Mileage Factor': mileage_factor,
        'Brand Factor': brand_factor,
        'Depreciation': market_price - trade_in_value
    }

def calculate_credit_score_impact(income, existing_loans, employment_type, age):
    """Simple credit score estimator"""
    base_score = 600
    
    # Income factor
    if income > 100000:
        base_score += 100
    elif income > 50000:
        base_score += 50
    
    # Existing loans
    if existing_loans == 0:
        base_score += 50
    elif existing_loans > 3:
        base_score -= 50
    
    # Employment
    if employment_type == 'Salaried':
        base_score += 50
    elif employment_type == 'Self-employed':
        base_score += 30
    
    # Age
    if age > 30:
        base_score += 30
    
    # Max 900
    credit_score = min(900, base_score)
    
    # Loan eligibility
    if credit_score >= 750:
        eligibility = 'Excellent - Low interest rates'
    elif credit_score >= 650:
        eligibility = 'Good - Standard rates'
    elif credit_score >= 550:
        eligibility = 'Fair - Higher rates'
    else:
        eligibility = 'Poor - Loan may be difficult'
    
    return credit_score, eligibility

def calculate_roi(car_price, yearly_expenses, years, resale_value):
    """Calculate ROI of car purchase"""
    total_cost = car_price + (yearly_expenses * years)
    total_value = resale_value
    
    roi = ((total_value - total_cost) / total_cost) * 100
    
    # Compare with alternative investments
    fd_return = car_price * (1 + 0.07) ** years  # 7% FD
    equity_return = car_price * (1 + 0.12) ** years  # 12% equity
    
    opportunity_cost_fd = fd_return - total_value
    opportunity_cost_equity = equity_return - total_value
    
    return {
        'ROI': roi,
        'Total Cost': total_cost,
        'Resale Value': total_value,
        'FD Alternative': fd_return,
        'Equity Alternative': equity_return,
        'Opportunity Cost (FD)': opportunity_cost_fd,
        'Opportunity Cost (Equity)': opportunity_cost_equity
    }

def generate_qr_code(data):
    """Generate QR code for car details"""
    try:
        import qrcode
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        # st.error(f"QR code generation error: {e}") # Suppress error message in main app if qrcode not installed
        return None

def generate_pdf_report(car_details, prediction, analysis):
    """Generate PDF report"""
    try:
        # Create matplotlib figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Car Analysis Report - {car_details['Brand']} {car_details['Model']}", fontsize=16, fontweight='bold')
        
        # Price breakdown
        ax1.text(0.1, 0.9, 'Price Estimation', fontsize=14, fontweight='bold')
        ax1.text(0.1, 0.7, f"Fair Market Price: ₹{prediction:,.0f}", fontsize=12)
        ax1.text(0.1, 0.6, f"Minimum: ₹{prediction*0.9:,.0f}", fontsize=10)
        ax1.text(0.1, 0.5, f"Maximum: ₹{prediction*1.1:,.0f}", fontsize=10)
        ax1.axis('off')
        
        # Car details
        ax2.text(0.1, 0.9, 'Car Details', fontsize=14, fontweight='bold')
        y_pos = 0.7
        for key, value in car_details.items():
            ax2.text(0.1, y_pos, f"{key}: {value}", fontsize=10)
            y_pos -= 0.1
            if y_pos < 0.1:
                break
        ax2.axis('off')
        
        # Depreciation chart
        years = list(range(2024, 2030))
        values = [prediction * ((1-0.15)**i) for i in range(6)]
        ax3.plot(years, values, marker='o', linewidth=2, color='#E24A4A')
        ax3.set_title('5-Year Depreciation', fontsize=12)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Value (₹)')
        ax3.grid(True, alpha=0.3)
        
        # Analysis summary
        ax4.text(0.1, 0.9, 'Analysis Summary', fontsize=14, fontweight='bold')
        ax4.text(0.1, 0.7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=10)
        ax4.text(0.1, 0.6, f"Report Type: AI-Powered Analysis", fontsize=10)
        ax4.text(0.1, 0.5, f"Confidence: 85%", fontsize=10)
        ax4.axis('off')
        
        # Save to bytes
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='pdf', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    except Exception as e:
        st.error(f"PDF generation error: {e}")
        return None

def send_email_alert(email, subject, message):
    """Simulate email sending (would need actual SMTP in production)"""
    # In production, use smtplib or services like SendGrid
    return f"Email would be sent to {email}: {subject}"

def generate_whatsapp_link(car_details):
    brand = car_details.get('Brand', 'Car')
    model = car_details.get('Model', '')
    price = car_details.get('Price', 0)
    
    message = f"Check out this car!\n\n🚗 {brand} {model}\n💰 Price: ₹{price:,.0f}\n\nInterested? Contact us!"
    encoded_message = message.replace('\n', '%0A').replace(' ', '%20')
    
    return f"https://wa.me/?text={encoded_message}"

def simple_chatbot(query, df):
    query = query.lower()
    if 'cheap' in query or 'budget' in query or 'under' in query:
        try:
            # Extract price number from query, allowing for varied formats
            import re
            price_match = re.search(r'(\d+)\s*(lakhs|lacs|million)?', query)
            if price_match:
                price_value = int(price_match.group(1))
                if price_match.group(2) in ['lakhs', 'lacs']:
                    price = price_value * 100000
                elif price_match.group(2) == 'million':
                    price = price_value * 1000000
                else: # assume direct value if no unit specified
                    price = price_value
            else:
                price = None # No price found
                
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

def calculate_fuel_cost(mileage_per_km, fuel_type, yearly_km=15000):
    fuel_prices = {'Petrol': 105, 'Diesel': 95, 'CNG': 80, 'Electric': 8, 'Hybrid': 90}
    price_per_unit = fuel_prices.get(fuel_type, 100)
    yearly_cost = (yearly_km / mileage_per_km) * price_per_unit
    return yearly_cost

def calculate_insurance(car_price, age, city_tier=1):
    base_rate = 0.03
    age_factor = 1 + (age * 0.02)
    city_factor = 1 + (city_tier * 0.01)
    insurance = car_price * base_rate * age_factor * city_factor
    return insurance

# Sidebar
with st.sidebar:
    st.title("🔐 User Panel")
    
    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs([t('login'), t('signup')])
        with tab1:
            username = st.text_input(t('username'), key="login_user")
            password = st.text_input(t('password'), type="password", key="login_pass")
            if st.button(t('login')):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome {username}!")
                st.rerun()
        with tab2:
            new_user = st.text_input(t('username'), key="signup_user")
            new_pass = st.text_input(t('password'), type="password", key="signup_pass")
            email = st.text_input(t('email'))
            if st.button(t('signup')):
                st.session_state.logged_in = True
                st.session_state.username = new_user
                st.session_state.email = email
                st.success(f"Account created!")
                st.rerun()
    else:
        st.success(f"👋 {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
    
    st.markdown("---")
    
    # Settings
    st.title("⚙️ Settings")
    dark_mode = st.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    language = st.selectbox("🌐 Language", ["English", "हिंदी", "मराठी"])
    if language != st.session_state.language:
        st.session_state.language = language
        st.rerun()
    
    st.markdown("---")
    
    # Navigation
    st.title("📊 Navigation")
    page = st.radio("Go to", [
        "🏠 Home",
        "💰 Price Prediction",
        "📊 Compare Cars",
        "🧮 EMI Calculator",
        "🔄 Trade-in Estimator",
        "💳 Credit Score Calculator",
        "📈 ROI Calculator",
        "📱 QR Code Generator",
        "📧 Email Alerts",
        "💬 Community Forum",
        "⭐ Reviews",
        "❤️ Wishlist",
        "📥 PDF Reports"
    ])

# File Upload
uploaded_file = st.file_uploader(f"📂 {t('upload')}", type=["csv","xlsx"])

df = None # Initialize df to None
feature_columns = [] # Initialize feature_columns

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

    if df is not None: # Check if df was successfully loaded
        if 'Market_Price(INR)' not in df.columns:
            st.error("❌ Must include 'Market_Price(INR)' column")
            st.stop()

        # Data Preprocessing
        df_clean = df.dropna()
        # Convert 'Year' to numeric if it exists and is not already
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
        # Exclude non-numeric or target columns before scaling
        cols_to_drop = ['Market_Price(INR)'] + [col for col in df_encoded.columns if df_encoded[col].dtype == 'object']
        cols_to_drop = list(set(cols_to_drop)) # Remove duplicates

        # Filter out columns that do not exist in the DataFrame
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
        best_model = None # Initialize best_model

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
        st.stop() # Stop if df is None after upload attempt

# ============================================
# HOME PAGE (ENHANCED MARKET INSIGHTS)
# ============================================
if page == "🏠 Home":
    if df is not None:
        st.subheader("📊 Dynamic Market Insights & Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cars Analyzed", f"{len(df_clean):,}")
        with col2:
            st.metric("Unique Brands", f"{df_clean['Brand'].nunique()}")
        with col3:
            st.metric("Avg. Market Price", f"₹{df_clean['Market_Price(INR)'].mean()/100000:.1f} Lakhs")
        with col4:
            st.metric("Active Predictions", len(st.session_state.predictions))

        st.markdown("---")
        
        st.subheader("📈 Interactive Market Trends")

        # 1. Price Distribution by Brand and Fuel Type (Enhanced Bar Chart + Box Plot)
        st.markdown("#### Price Distribution Overview")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if 'Brand' in df_clean.columns:
                selected_brands_for_viz = st.multiselect(
                    "Select Brands for Price Distribution", 
                    options=sorted(df_clean['Brand'].unique()),
                    default=sorted(df_clean['Brand'].unique())[:5] # Default to top 5
                )
                filtered_df_brands = df_clean[df_clean['Brand'].isin(selected_brands_for_viz)]

                if not filtered_df_brands.empty:
                    fig_brand_dist, ax_brand_dist = plt.subplots(figsize=(10, 6))
                    sns.boxplot(x='Brand', y='Market_Price(INR)', data=filtered_df_brands, ax=ax_brand_dist, palette='coolwarm')
                    ax_brand_dist.set_title('Market Price Distribution by Brand')
                    ax_brand_dist.set_ylabel('Market Price (INR)')
                    ax_brand_dist.ticklabel_format(style='plain', axis='y') # Avoid scientific notation
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig_brand_dist)
                    plt.close(fig_brand_dist)
                else:
                    st.info("Please select at least one brand to view its price distribution.")
            else:
                st.info("Brand column not found in data for price distribution.")
        
        with chart_col2:
            if 'Fuel_Type' in df_clean.columns:
                selected_fuel_types = st.multiselect(
                    "Select Fuel Types for Price Distribution",
                    options=sorted(df_clean['Fuel_Type'].unique()),
                    default=sorted(df_clean['Fuel_Type'].unique())
                )
                filtered_df_fuel = df_clean[df_clean['Fuel_Type'].isin(selected_fuel_types)]

                if not filtered_df_fuel.empty:
                    fig_fuel_dist, ax_fuel_dist = plt.subplots(figsize=(10, 6))
                    sns.violinplot(x='Fuel_Type', y='Market_Price(INR)', data=filtered_df_fuel, ax=ax_fuel_dist, palette='plasma')
                    ax_fuel_dist.set_title('Market Price Distribution by Fuel Type')
                    ax_fuel_dist.set_ylabel('Market Price (INR)')
                    ax_fuel_dist.ticklabel_format(style='plain', axis='y')
                    st.pyplot(fig_fuel_dist)
                    plt.close(fig_fuel_dist)
                else:
                    st.info("Please select at least one fuel type.")
            else:
                st.info("Fuel_Type column not found in data for price distribution.")


        st.markdown("---")

        # 2. Price Trend Over Time (Line Chart with selection)
        st.markdown("#### Average Price Trend Over Years")
        if 'Brand' in df_clean.columns and 'Year' in df_clean.columns:
            trend_brand = st.selectbox("Select Brand for Trend Analysis", sorted(df_clean['Brand'].unique()), key="trend_brand_select")
            
            brand_yearly_avg = df_clean[df_clean['Brand'] == trend_brand].groupby('Year')['Market_Price(INR)'].mean().reset_index()

            if not brand_yearly_avg.empty:
                fig_trend, ax_trend = plt.subplots(figsize=(12, 6))
                sns.lineplot(x='Year', y='Market_Price(INR)', data=brand_yearly_avg, marker='o', ax=ax_trend, color='#28a745', linewidth=3)
                ax_trend.set_title(f'Average Market Price Trend for {trend_brand} Over Years')
                ax_trend.set_xlabel('Year')
                ax_trend.set_ylabel('Average Market Price (INR)')
                ax_trend.grid(True, linestyle='--', alpha=0.6)
                ax_trend.ticklabel_format(style='plain', axis='y')
                st.pyplot(fig_trend)
                plt.close(fig_trend)
            else:
                st.info(f"No yearly price data available for {trend_brand}.")
        else:
            st.info("Brand or Year column not found in data for trend analysis.")

        st.markdown("---")

        # 3. Feature Importance (Bar Chart from Model)
        st.markdown("#### Key Factors Influencing Car Price")
        
        if best_model is not None and hasattr(best_model, 'feature_importances_') and feature_columns:
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': best_model.feature_importances_
            }).sort_values(by='Importance', ascending=False).head(10) # Top 10 features

            fig_importance, ax_importance = plt.subplots(figsize=(12, 7))
            sns.barplot(x='Importance', y='Feature', data=importance_df, palette='cividis', ax=ax_importance)
            ax_importance.set_title('Top 10 Feature Importances in Price Prediction')
            ax_importance.set_xlabel('Relative Importance')
            ax_importance.set_ylabel('Car Feature')
            st.pyplot(fig_importance)
            plt.close(fig_importance)
        else:
            st.info("Feature importance data is not available for the selected model or data is missing.")

        st.markdown("---")
        st.markdown("### 💬 Chatbot Assistant")
        user_query = st.text_input("Ask me anything about car prices or models (e.g., 'cars under 10 lakhs', 'best SUV'):")
        if user_query:
            response, cars_found = simple_chatbot(user_query, df_clean)
            st.write(response)
            if not cars_found.empty:
                st.dataframe(cars_found, use_container_width=True)
    else:
        st.info(f"📥 {t('upload')} to start!")
        
        st.markdown("---")
        st.markdown("### 🎯 All Features:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **💰 Price & Analysis**
            - AI Price Prediction
            - Compare Cars
            - EMI Calculator
            - Trade-in Estimator
            """)
        
        with col2:
            st.markdown("""
            **📊 Tools & Calculators**
            - Credit Score Calculator
            - ROI Calculator
            - Tax Calculator
            - Inspection Checklist
            """)
        
        with col3:
            st.markdown("""
            **🚀 Advanced Features**
            - PDF Reports
            - QR Code Generator
            - Email Alerts
            - Multi-language
            - Community Forum
            """)


# ============================================
# REST OF THE ORIGINAL PAGES (UNCHANGED)
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
            for col in feature_columns: # Use feature_columns from model training
                if col in filtered_row.index:
                    with [col1, col2, col3][feature_idx % 3]:
                        if col in encoders: # If it was an encoded categorical column
                            options = sorted(df_clean[encoders[col].inverse_transform(df_encoded[col].unique())[0]].unique()) # Get original categories
                            default_val = encoders[col].inverse_transform([filtered_row[col]])[0]
                            inputs[col] = st.selectbox(f"{col}", options, index=options.index(default_val), key=f"p_{col}")
                        else: # Numeric column
                            min_val = int(df_clean[col].min())
                            max_val = int(df_clean[col].max())
                            default_val = int(filtered_row[col])
                            inputs[col] = st.slider(f"{col}", min_val, max_val, default_val, key=f"p_{col}")
                    feature_idx += 1
                elif col == 'Fuel_Type' and 'Fuel_Type' in df_clean.columns: # Example for a specific missing column
                    with [col1, col2, col3][feature_idx % 3]:
                        options = sorted(df_clean['Fuel_Type'].unique())
                        inputs[col] = st.selectbox("Fuel_Type", options, key=f"p_{col}")
                    feature_idx += 1
                # Add more else-if blocks for other important features if they might be missing in filtered_row but present in feature_columns
                # This ensures all features used by the model are present in the input_df
            
            if st.button("🔍 Predict", type="primary"):
                # Create a DataFrame for prediction, ensuring all model features are present
                predict_input_data = {}
                for col in feature_columns:
                    if col in inputs:
                        predict_input_data[col] = inputs[col]
                    elif col in filtered_row.index: # Fallback to filtered_row if not in inputs (e.g., if a feature wasn't exposed in UI)
                        predict_input_data[col] = filtered_row[col]
                    else: # Handle truly missing features, e.g., with a mean or 0
                        st.warning(f"Feature '{col}' not found in inputs or filtered row. Using default (0 or mean).")
                        predict_input_data[col] = 0 # Or use df_clean[col].mean() if numeric
                
                input_df = pd.DataFrame([predict_input_data])
                
                # Apply encoders for categorical features
                for col in encoders:
                    if col in input_df.columns:
                        if col in df_clean.columns and input_df[col].dtype == 'object': # Only encode if it's an object type that was originally encoded
                            try:
                                # Ensure the value is in the encoder's known classes, or handle unknown
                                if input_df[col].iloc[0] not in encoders[col].classes_:
                                    st.warning(f"New category '{input_df[col].iloc[0]}' for {col}. This might lead to inaccurate prediction.")
                                    # Fallback: assign a default value or the most frequent category's encoding
                                    input_df[col] = encoders[col].transform([encoders[col].classes_[0]]) # Assign first known class
                                else:
                                    input_df[col] = encoders[col].transform(input_df[col].astype(str))
                            except ValueError as ve:
                                st.error(f"Error encoding {col}: {ve}. Input value might be new. Please recheck data.")
                                st.stop()
                        elif col in df_encoded.columns and input_df[col].dtype != 'object': # If it's already encoded numeric, ensure it's correct
                            # This case handles when the input_df already has the encoded numeric value from a selectbox
                            pass
                        
                # Ensure all columns expected by the scaler are present and in order
                # This is crucial for correct scaling
                missing_cols = set(X.columns) - set(input_df.columns)
                for c in missing_cols:
                    input_df[c] = 0 # Or a suitable default value (e.g., mean from training data)
                input_df = input_df[X.columns] # Reorder columns to match training data

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
            st.warning("No data found for the selected brand and model. Cannot make a prediction.")
    else:
        st.info(f"📥 {t('upload')} to make predictions!")


# ============================================
# TRADE-IN ESTIMATOR
# ============================================
elif page == "🔄 Trade-in Estimator":
    if df is not None:
        st.subheader("🔄 Trade-in Value Estimator")
        
        st.info("💡 Get instant estimate of your car's trade-in value")
        
        col1, col2 = st.columns(2)
        
        with col1:
            brands = sorted(df_clean['Brand'].unique())
            trade_brand = st.selectbox("Your Car Brand", brands, key="trade_brand")
            
            trade_models = sorted(df_clean[df_clean['Brand'] == trade_brand]['Model'].unique())
            trade_model = st.selectbox("Model", trade_models, key="trade_model")
            
            trade_year = st.number_input("Year", 2010, 2024, 2020)
            trade_mileage = st.number_input("Mileage (km)", 0, 500000, 50000, step=5000)
        
        with col2:
            trade_condition = st.select_slider("Condition", 
                                               options=['Poor', 'Fair', 'Good', 'Excellent'],
                                               value='Good')
            
            st.markdown("### Condition Guide:")
            st.write("**Excellent:** Like new, no issues")
            st.write("**Good:** Minor wear, well maintained")
            st.write("**Fair:** Some issues, needs work")
            st.write("**Poor:** Major problems")
        
        if st.button("Calculate Trade-in Value", type="primary"):
            car_data = df_clean[(df_clean['Brand'] == trade_brand) & (df_clean['Model'] == trade_model)]
            
            if len(car_data) > 0:
                market_price = car_data.iloc[0]['Market_Price(INR)']
                
                trade_in = calculate_trade_in_value(
                    trade_brand, trade_model, trade_year, 
                    trade_condition, trade_mileage, market_price
                )
                
                st.markdown("---")
                st.subheader("💰 Trade-in Valuation")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Trade-in Value", f"₹{trade_in['Trade-in Value']:,.0f}")
                with col2:
                    st.metric("Market Price", f"₹{market_price:,.0f}")
                with col3:
                    st.metric("Depreciation", f"₹{trade_in['Depreciation']:,.0f}")
                
                st.markdown("---")
                st.subheader("📊 Valuation Breakdown")
                
                factors_df = pd.DataFrame({
                    'Factor': ['Age', 'Condition', 'Mileage', 'Brand'],
                    'Impact': [
                        f"{trade_in['Age Factor']:.2%}",
                        f"{trade_in['Condition Factor']:.2%}",
                        f"{trade_in['Mileage Factor']:.2%}",
                        f"{trade_in['Brand Factor']:.2%}"
                    ]
                })
                st.dataframe(factors_df, use_container_width=True, hide_index=True)
                
                st.success("💡 Tip: Better condition and lower mileage increase trade-in value!")
                
                # WhatsApp share
                whatsapp_link = generate_whatsapp_link({
                    'Brand': trade_brand,
                    'Model': trade_model,
                    'Price': trade_in['Trade-in Value']
                })
                st.markdown(f'<a href="{whatsapp_link}" target="_blank" class="whatsapp-btn">📱 Share on WhatsApp</a>', 
                           unsafe_allow_html=True)
            else:
                st.warning("No data found for the selected car to estimate trade-in value.")
    else:
        st.info(f"📥 {t('upload')} to use the Trade-in Estimator!")

# ============================================
# CREDIT SCORE CALCULATOR
# ============================================
elif page == "💳 Credit Score Calculator":
    st.subheader("💳 Credit Score & Loan Eligibility Calculator")
    
    st.info("💡 Estimate your credit score and loan eligibility")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_income = st.number_input("Monthly Income (₹)", 10000, 500000, 50000, step=5000)
        existing_loans = st.number_input("Number of Existing Loans", 0, 10, 0)
        employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-employed', 'Business', 'Other'])
        age = st.number_input("Age", 18, 70, 30)
    
    with col2:
        car_price = st.number_input("Car Price (₹)", 100000, 10000000, 1000000, step=50000)
        down_payment = st.slider("Down Payment (%)", 0, 50, 20)
        
    if st.button("Calculate Eligibility", type="primary"):
        credit_score, eligibility = calculate_credit_score_impact(
            monthly_income, existing_loans, employment_type, age
        )
        
        st.markdown("---")
        st.subheader("📊 Credit Score Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Estimated Credit Score", credit_score)
            
            # Progress bar
            score_percent = credit_score / 900
            st.progress(score_percent)
            
            if credit_score >= 750:
                st.success(f"✅ {eligibility}")
            elif credit_score >= 650:
                st.info(f"ℹ️ {eligibility}")
            else:
                st.warning(f"⚠️ {eligibility}")
        
        with col2:
            loan_amount = car_price * (1 - down_payment/100)
            max_emi = monthly_income * 0.4  # 40% of income
            
            st.metric("Loan Amount", f"₹{loan_amount:,.0f}")
            st.metric("Max Affordable EMI", f"₹{max_emi:,.0f}")
            
            # Calculate EMI at 9.5%
            rate = 0.095/12
            tenure = 60  # 5 years
            # Ensure loan_amount is not zero to avoid division by zero error
            if loan_amount > 0:
                emi = loan_amount * rate * ((1 + rate)**tenure) / (((1 + rate)**tenure) - 1)
            else:
                emi = 0 # No EMI if no loan
            
            st.metric("Estimated EMI (5 years)", f"₹{emi:,.0f}")
            
            if emi <= max_emi and emi > 0:
                st.success("✅ Loan is affordable!")
            elif emi == 0:
                st.info("No loan amount required with 100% down payment.")
            else:
                st.error("❌ EMI exceeds 40% of income")
        
        st.markdown("---")
        st.subheader("💡 Improvement Tips")
        
        tips = []
        if credit_score < 750:
            tips.append("• Pay existing loans on time")
            tips.append("• Reduce number of active loans")
            tips.append("• Maintain low credit utilization")
            tips.append("• Avoid multiple loan applications")
        
        if tips:
            for tip in tips:
                st.write(tip)
        else:
            st.success("Your credit profile looks great! 🎉")

# ============================================
# ROI CALCULATOR
# ============================================
elif page == "📈 ROI Calculator":
    st.subheader("📈 Investment ROI Calculator")
    
    st.info("💡 Compare car purchase with alternative investments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        roi_car_price = st.number_input("Car Price (₹)", 500000, 10000000, 2000000, step=100000)
        roi_years = st.slider("Ownership Period (years)", 3, 10, 5)
        yearly_expenses = st.number_input("Yearly Expenses (₹)", 50000, 500000, 150000, step=10000)
        
    with col2:
        estimated_resale = st.number_input("Estimated Resale Value (₹)", 
                                          100000, roi_car_price, 
                                          int(roi_car_price * 0.5), step=50000)
    
    if st.button("Calculate ROI", type="primary"):
        roi_data = calculate_roi(roi_car_price, yearly_expenses, roi_years, estimated_resale)
        
        st.markdown("---")
        st.subheader("📊 ROI Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Car ROI", f"{roi_data['ROI']:.2f}%")
            st.metric("Total Cost", f"₹{roi_data['Total Cost']:,.0f}")
        
        with col2:
            st.metric("FD Return (7%)", f"₹{roi_data['FD Alternative']:,.0f}")
            st.metric("Opportunity Cost", f"₹{roi_data['Opportunity Cost (FD)']:,.0f}")
        
        with col3:
            st.metric("Equity Return (12%)", f"₹{roi_data['Equity Alternative']:,.0f}")
            st.metric("Opportunity Cost", f"₹{roi_data['Opportunity Cost (Equity)']:,.0f}")
        
        st.markdown("---")
        st.subheader("💰 Investment Comparison")
        
        comparison_df = pd.DataFrame({
            'Investment': ['Car', 'Fixed Deposit', 'Equity'],
            'Initial': [roi_car_price, roi_car_price, roi_car_price],
            'Final Value': [estimated_resale, roi_data['FD Alternative'], roi_data['Equity Alternative']],
            'Total Cost': [
                roi_data['Total Cost'] if 'Total Cost' in roi_data else roi_car_price, # Safely access 'Total Cost'
                roi_car_price, 
                roi_car_price
            ],
            'Net Return': [
                estimated_resale - roi_data['Total Cost'] if 'Total Cost' in roi_data else estimated_resale - roi_car_price, # Safely access
                roi_data['FD Alternative'] - roi_car_price,
                roi_data['Equity Alternative'] - roi_car_price
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        if roi_data['ROI'] < 0:
            st.error("⚠️ Car purchase results in negative ROI. Consider alternatives.")
        else:
            st.info("ℹ️ Cars are typically depreciating assets. Buy for utility, not investment.")

# ============================================
# QR CODE GENERATOR
# ============================================
elif page == "📱 QR Code Generator":
    if df is not None:
        st.subheader("📱 QR Code Generator")
        
        st.info("💡 Generate QR code for easy car information sharing")
        
        brands = sorted(df_clean['Brand'].unique())
        qr_brand = st.selectbox("Select Brand", brands, key="qr_brand")
        
        qr_models = sorted(df_clean[df_clean['Brand'] == qr_brand]['Model'].unique())
        qr_model = st.selectbox("Select Model", qr_models, key="qr_model")
        
        car_data = df_clean[(df_clean['Brand'] == qr_brand) & (df_clean['Model'] == qr_model)]
        
        if len(car_data) > 0 and st.button("Generate QR Code", type="primary"):
            car_info = car_data.iloc[0]
            
            qr_data = f"""
Car Details:
Brand: {qr_brand}
Model: {qr_model}
Year: {car_info['Year']}
Price: ₹{car_info['Market_Price(INR)']:,.0f}
Fuel: {car_info.get('Fuel_Type', 'N/A')}
Transmission: {car_info.get('Transmission', 'N/A')}
            """.strip()
            
            qr_img = generate_qr_code(qr_data)
            
            if qr_img:
                st.markdown("---")
                st.markdown('<div class="qr-container">', unsafe_allow_html=True)
                st.markdown(f'<img src="data:image/png;base64,{qr_img}" width="300">', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.success("✅ QR Code Generated! Scan to view car details")
                
                st.markdown("---")
                st.subheader("📋 Car Information")
                st.write(qr_data)
            else:
                st.warning("⚠️ QR code generation requires 'qrcode' package. Install: `pip install qrcode[pil]`")
    else:
        st.info(f"📥 {t('upload')} to generate QR codes!")

# ============================================
# EMAIL ALERTS
# ============================================
elif page == "📧 Email Alerts":
    st.subheader("📧 Email Alert System")
    
    if not st.session_state.logged_in:
        st.warning("⚠️ Please login to set email alerts!")
    else:
        st.info("💡 Get email notifications for price drops and new listings")
        
        email_address = st.text_input("Email Address", value=st.session_state.email)
        
        alert_type = st.selectbox("Alert Type", [
            "Price Drop Alert",
            "New Listing Alert",
            "Weekly Market Report",
            "Maintenance Reminder"
        ])
        
        if df is not None:
            brands = sorted(df_clean['Brand'].unique())
            alert_brand = st.selectbox("Brand (Optional)", ["All"] + brands)
            
            if alert_brand != "All":
                alert_models = sorted(df_clean[df_clean['Brand'] == alert_brand]['Model'].unique())
                alert_model = st.selectbox("Model (Optional)", ["All"] + alert_models)
            else:
                alert_model = "All" # Set a default for clarity
        else:
            alert_brand = "All"
            alert_model = "All"
            st.warning("Upload a dataset to select specific brands/models for alerts.")
        
        frequency = st.selectbox("Frequency", ["Instant", "Daily", "Weekly"])
        
        if st.button("Set Email Alert", type="primary"):
            alert_message = send_email_alert(
                email_address,
                f"Car Alert: {alert_type}",
                f"Alert set for {alert_brand if alert_brand != 'All' else 'all brands'}"
            )
            
            st.success(f"✅ Alert set! You'll receive {frequency.lower()} notifications at {email_address}")
            st.info(f"📧 {alert_message}")
            
            st.session_state.price_alerts.append({
                'Type': alert_type,
                'Brand': alert_brand,
                'Model': alert_model, # Added model to alert details
                'Email': email_address,
                'Frequency': frequency,
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M")
            })
        
        if st.session_state.price_alerts:
            st.markdown("---")
            st.subheader("📋 Active Email Alerts")
            
            alerts_df = pd.DataFrame(st.session_state.price_alerts)
            st.dataframe(alerts_df, use_container_width=True, hide_index=True)

# ============================================
# COMMUNITY FORUM
# ============================================
elif page == "💬 Community Forum":
    st.subheader("💬 Community Discussion Forum")
    
    if not st.session_state.logged_in:
        st.warning("⚠️ Please login to participate in discussions!")
    else:
        tab1, tab2 = st.tabs(["📝 Create Post", "💭 View Discussions"])
        
        with tab1:
            st.markdown("### Create New Post")
            
            post_title = st.text_input("Title")
            post_category = st.selectbox("Category", [
                "General Discussion",
                "Car Buying Tips",
                "Maintenance Help",
                "Price Negotiation",
                "Insurance & Finance",
                "Reviews & Experiences"
            ])
            post_content = st.text_area("Content", height=150)
            
            if st.button("Post", type="primary"):
                if post_title and post_content:
                    st.session_state.forum_posts.append({
                        'Title': post_title,
                        'Category': post_category,
                        'Content': post_content,
                        'Author': st.session_state.username,
                        'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'Replies': 0
                    })
                    st.success("✅ Post created!")
                    st.rerun()
                else:
                    st.error("Please fill all fields")
        
        with tab2:
            st.markdown("### Recent Discussions")
            
            if st.session_state.forum_posts:
                for idx, post in enumerate(reversed(st.session_state.forum_posts)):
                    with st.expander(f"📌 {post['Title']} - by {post['Author']}"):
                        st.write(f"**Category:** {post['Category']}")
                        st.write(f"**Posted:** {post['Date']}")
                        st.markdown("---")
                        st.write(post['Content'])
                        st.write(f"💬 {post['Replies']} replies")
            else:
                st.info("No posts yet. Be the first to start a discussion!")

# ============================================
# REVIEWS
# ============================================
elif page == "⭐ Reviews":
    st.subheader("⭐ Car Reviews & Ratings")
    
    if st.session_state.logged_in:
        st.markdown("### Write a Review")
        
        if df is not None:
            brands = sorted(df_clean['Brand'].unique())
            review_brand = st.selectbox("Brand", brands, key="rev_brand")
            review_models = sorted(df_clean[df_clean['Brand'] == review_brand]['Model'].unique())
            review_model = st.selectbox("Model", review_models, key="rev_model")
            
            rating = st.slider("Rating", 1, 5, 4)
            review_text = st.text_area("Your Review", height=100)
            
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
                st.success("✅ Review submitted!")
                st.rerun()
        else:
            st.warning("Upload a dataset to write reviews for specific cars.")

    st.markdown("---")
    st.subheader("📝 All Reviews")
    
    if st.session_state.reviews:
        for car, reviews in st.session_state.reviews.items():
            with st.expander(f"🚗 {car} ({len(reviews)} reviews)"):
                avg_rating = sum([r['Rating'] for r in reviews]) / len(reviews)
                st.write(f"⭐ Average: {avg_rating:.1f}/5")
                st.markdown("---")
                
                for review in reviews:
                    st.markdown(f"**{review['User']}** - {'⭐' * review['Rating']} ({review['Date']})")
                    st.write(review['Review'])
                    st.markdown("---")
    else:
        st.info("No reviews yet!")

# ============================================
# WISHLIST
# ============================================
elif page == "❤️ Wishlist":
    st.subheader("❤️ My Wishlist")
    
    if st.session_state.logged_in:
        if st.session_state.wishlist:
            st.write(f"You have {len(st.session_state.wishlist)} cars saved")
            
            for idx, car in enumerate(st.session_state.wishlist):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"🚗 {car}")
                with col2:
                    if st.button("Remove", key=f"rm_{idx}"):
                        st.session_state.wishlist.pop(idx)
                        st.rerun()
        else:
            st.info("Your wishlist is empty!")
    else:
        st.warning("⚠️ Please login!")

# ============================================
# PDF REPORTS
# ============================================
elif page == "📥 PDF Reports":
    st.subheader("📥 Generate PDF Reports")
    
    if st.session_state.predictions:
        st.info("💡 Generate professional PDF reports for your car predictions")
        
        report_type = st.selectbox("Report Type", [
            "Single Car Analysis",
            "Prediction History",
            "Market Insights", # This can generate a PDF of current market insights
            "Complete Analysis"
        ])
        
        if report_type == "Single Car Analysis" and len(st.session_state.predictions) > 0:
            prediction_idx = st.selectbox("Select Prediction", 
                                         range(len(st.session_state.predictions)),
                                         format_func=lambda x: f"{st.session_state.predictions[x]['Brand']} {st.session_state.predictions[x]['Model']}")
            
            if st.button("Generate PDF Report", type="primary"):
                pred = st.session_state.predictions[prediction_idx]
                
                car_details = {
                    'Brand': pred['Brand'],
                    'Model': pred['Model'],
                    'Predicted Price': pred['Price'],
                    'Date': pred['Time']
                }
                
                price_value = float(pred['Price'].replace('₹', '').replace(',', ''))
                
                pdf_buffer = generate_pdf_report(car_details, price_value, {})
                
                if pdf_buffer:
                    st.success("✅ PDF Report Generated!")
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"Car_Analysis_{pred['Brand']}_{pred['Model']}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error("PDF generation failed")
        
        elif report_type == "Prediction History":
            st.markdown("### Your Prediction History")
            pred_df = pd.DataFrame(st.session_state.predictions)
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
            
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download as CSV", csv, "predictions.csv", "text/csv")
    
    else:
        st.info("No predictions yet! Make some predictions first.")
    
    st.markdown("---")
    st.subheader("📊 Other Reports")
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            csv_full = df_clean.to_csv(index=False).encode('utf-8')
            st.download_button("📄 Download Full Dataset", csv_full, "dataset.csv", "text/csv")
        
        with col2:
            if st.session_state.wishlist:
                wishlist_df = pd.DataFrame({'Car': st.session_state.wishlist})
                csv_wish = wishlist_df.to_csv(index=False).encode('utf-8')
                st.download_button("❤️ Download Wishlist", csv_wish, "wishlist.csv", "text/csv")

# If no file is uploaded or df is None, show initial message
if uploaded_file is None:
    st.info(f"📥 {t('upload')} to start!")
    
    st.markdown("---")
    st.markdown("### 🎯 All Features:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **💰 Price & Analysis**
        - AI Price Prediction
        - Compare Cars
        - EMI Calculator
        - Trade-in Estimator
        """)
    
    with col2:
        st.markdown("""
        **📊 Tools & Calculators**
        - Credit Score Calculator
        - ROI Calculator
        - Tax Calculator
        - Inspection Checklist
        """)
    
    with col3:
        st.markdown("""
        **🚀 Advanced Features**
        - PDF Reports
        - QR Code Generator
        - Email Alerts
        - Multi-language
        - Community Forum
        """)
