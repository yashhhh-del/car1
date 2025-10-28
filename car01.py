# ======================================================
# SMART CAR PRICING SYSTEM - INR PRICE PREDICTION
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
import joblib
import os

# ========================================
# MASTER CAR CATALOG (Luxury + All Brands)
# ========================================
CAR_CATALOG = {
    "Maruti Suzuki": ["Swift", "Dzire", "Baleno", "WagonR", "Alto", "Brezza", "Ertiga", "Ciaz", "S-Presso", "Celerio"],
    "Hyundai": ["Creta", "i20", "Verna", "Venue", "i10", "Santro", "Tucson", "Alcazar", "Kona"],
    "Tata": ["Nexon", "Harrier", "Altroz", "Tiago", "Punch", "Safari", "Tigor", "Nexon EV"],
    "Mahindra": ["Thar", "XUV700", "Scorpio", "Bolero", "XUV300", "XUV500"],
    "Toyota": ["Innova Crysta", "Fortuner", "Camry", "Glanza", "Urban Cruiser"],
    "Honda": ["City", "Amaze", "WR-V", "Jazz", "Civic"],
    "Kia": ["Seltos", "Sonet", "Carens", "Carnival"],
    "BMW": ["3 Series", "5 Series", "X1", "X3", "X5", "X7", "M3", "M5", "i4", "iX", "7 Series"],
    "Mercedes-Benz": ["C-Class", "E-Class", "S-Class", "GLC", "GLE", "GLS", "A-Class", "AMG GT", "G-Class"],
    "Audi": ["A4", "A6", "Q3", "Q5", "Q7", "Q8", "A8", "RS Q8", "e-tron"],
    "Porsche": ["911", "Cayenne", "Macan", "Panamera", "Taycan"],
    "Lamborghini": ["Huracan", "Urus", "Aventador"],
    "Ferrari": ["488", "Roma", "SF90", "Portofino"],
    "Rolls-Royce": ["Phantom", "Ghost", "Cullinan", "Wraith"],
    "Tesla": ["Model 3", "Model Y", "Model S", "Model X"],
}

# ========================================
# INITIALIZE SESSION STATE
# ========================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_ok' not in st.session_state:
        st.session_state.model_ok = False
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = pd.DataFrame()

# ========================================
# DATA LOADING FUNCTIONS
# ========================================

@st.cache_data
def load_data(file):
    """Load and clean CSV data - FOCUS ON INR PRICE PREDICTION"""
    df = pd.read_csv(file)
    
    st.info(f"📁 Original columns: {list(df.columns)}")
    
    # FIND PRICE COLUMN AUTOMATICALLY
    price_col = None
    price_keywords = ['price', 'amount', 'cost', 'value', 'inr', 'rs', 'rupee']
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in price_keywords):
            price_col = col
            st.success(f"✅ Price column found: {price_col}")
            break
    
    if price_col:
        df = df.rename(columns={price_col: 'Market_Price_INR'})
    else:
        # If no price column found, use first numeric column as price
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            price_col = numeric_cols[0]
            df = df.rename(columns={price_col: 'Market_Price_INR'})
            st.warning(f"⚠ Using first numeric column as price: {price_col}")
        else:
            st.error("❌ No price column found in CSV!")
            return pd.DataFrame()
    
    # STANDARDIZE COLUMN NAMES FOR PREDICTION
    rename_map = {
        'brand': 'Brand', 'model': 'Model', 'year': 'Year', 
        'mileage': 'Mileage', 'fuel': 'Fuel_Type', 
        'transmission': 'Transmission', 'city': 'City',
        'company': 'Brand', 'car_name': 'Model', 'kms_driven': 'Mileage'
    }
    
    for old, new in rename_map.items():
        for col in df.columns:
            if old in col.lower() and col != new:
                df = df.rename(columns={col: new})
                st.info(f"🔄 Renamed '{col}' → '{new}'")
                break
    
    # CLEAN DATA FOR PRICE PREDICTION
    original_rows = len(df)
    
    # Remove rows with missing price
    df = df.dropna(subset=['Market_Price_INR'])
    
    # Convert price to numeric
    df['Market_Price_INR'] = pd.to_numeric(df['Market_Price_INR'], errors='coerce')
    df = df.dropna(subset=['Market_Price_INR'])
    
    # Clean year column
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year'])
        df['Year'] = df['Year'].astype(int)
    
    # Clean mileage column  
    if 'Mileage' in df.columns:
        df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
        df = df.dropna(subset=['Mileage'])
    
    st.success(f"✅ Data cleaned: {original_rows} → {len(df)} rows")
    st.info(f"💰 Price range: ₹{df['Market_Price_INR'].min():,} to ₹{df['Market_Price_INR'].max():,}")
    
    return df

# ========================================
# MODEL TRAINING FOR INR PRICE PREDICTION
# ========================================

@st.cache_resource
def train_model(df):
    """Train model to predict INR price from your data"""
    current_year = datetime.now().year
    df_model = df.copy()
    
    st.write("🔧 Preparing features for price prediction...")
    
    # FEATURE ENGINEERING FOR BETTER PRICE PREDICTION
    if 'Year' in df_model.columns:
        df_model['Car_Age'] = current_year - df_model['Year']
        st.info(f"✅ Added feature: Car_Age (Range: {df_model['Car_Age'].min()} - {df_model['Car_Age'].max()} years)")
    
    if 'Brand' in df_model.columns:
        brand_avg = df_model.groupby('Brand')['Market_Price_INR'].mean()
        df_model['Brand_Avg_Price'] = df_model['Brand'].map(brand_avg)
        st.info(f"✅ Added feature: Brand_Avg_Price ({len(brand_avg)} brands)")
    
    # ENCODE CATEGORICAL VARIABLES
    cat_cols = df_model.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le
        st.info(f"✅ Encoded: {col} ({len(le.classes_)} categories)")

    # PREPARE FEATURES AND TARGET (INR PRICE)
    X = df_model.drop(columns=['Market_Price_INR'], errors='ignore')
    y = df_model['Market_Price_INR']  # THIS IS WHAT WE PREDICT!
    
    st.write(f"🎯 Predicting: Market_Price_INR (Target)")
    st.write(f"📊 Using {X.shape[1]} features for prediction")
    
    # Scale features
    X_scaled = StandardScaler().fit_transform(X)
    
    # HYPERPARAMETER TUNING FOR BETTER PRICE PREDICTION
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5]
    }
    
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1), 
        param_grid, 
        cv=5,
        scoring='r2'
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    with st.spinner('🤖 Training model to predict your INR prices...'):
        grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    # COMPREHENSIVE MODEL EVALUATION
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='r2')
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # DISPLAY RESULTS
    st.success(f"""
    🎯 **INR Price Prediction Model Ready!**
    
    **Performance Metrics:**
    - R² Score: {r2:.3f} ({r2*100:.1f}% variance explained)
    - Mean Absolute Error: ₹{mae:,.0f}
    - Mean Absolute % Error: {mape:.1f}%
    - RMSE: ₹{rmse:,.0f}
    - Cross-val Consistency: {cv_scores.mean()*100:.1f}%
    
    **Best Parameters:** {grid.best_params_}
    """)
    
    return {
        'model': best_model, 
        'scaler': StandardScaler().fit(X), 
        'encoders': encoders, 
        'features': X.columns.tolist(),
        'r2': r2, 
        'accuracy': r2 * 100, 
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'cv_mean': cv_scores.mean() * 100,
        'importances': importances,
        'best_params': grid.best_params_,
        'feature_names': X.columns.tolist()
    }

# ========================================
# PRICE PREDICTION FUNCTION
# ========================================

def predict_inr_price(model_data, input_data, df_clean):
    """Predict INR price using trained model"""
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Apply same encoding as training
        for col in model_data['encoders']:
            if col in input_df.columns:
                try: 
                    input_df[col] = model_data['encoders'][col].transform([input_data[col]])[0]
                except:
                    # If new category, use most common value
                    input_df[col] = 0
        
        # Ensure all features are present
        for col in model_data['features']:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select and order features correctly
        input_df = input_df[model_data['features']]
        
        # Scale features
        input_scaled = model_data['scaler'].transform(input_df)
        
        # Predict price
        predicted_price = model_data['model'].predict(input_scaled)[0]
        
        return predicted_price, "AI Model"
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return None, "Error"

# ========================================
# WEB FALLBACK FUNCTION
# ========================================

@st.cache_data(ttl=3600)
def get_web_price(brand, model, year=None, city="Delhi"):
    """Get car price from web sources as fallback"""
    query = f"{brand.replace(' ', '-')}-{model.replace(' ', '-')}".lower()
    if year: 
        query += f"-{year}"
    
    urls = [
        f"https://www.cardekho.com/used-{query}+in+{city.lower()}",
        f"https://www.carwale.com/used/cars-in-{city.lower()}/search/?query={query}"
    ]
    
    headers = {"User-Agent": "Mozilla/5.0"}
    prices = []
    
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code != 200: 
                continue
                
            soup = BeautifulSoup(r.text, 'html.parser')
            texts = soup.find_all(string=re.compile(r'₹'))
            
            for t in texts:
                m = re.search(r'₹\s*([\d,.]+)\s*(lakh|crore)?', t, re.I)
                if m:
                    val = float(m.group(1).replace(',', ''))
                    if m.group(2) and 'crore' in m.group(2).lower(): 
                        val *= 100
                    prices.append(int(val * 100000))
                    
        except Exception as e:
            continue
            
    return int(np.mean(prices)) if prices else None

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    # Initialize session state
    initialize_session_state()
    
    # Page config
    st.set_page_config(
        page_title="Smart Car Pricing", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    st.title("🚗 Smart Car Pricing System")
    st.markdown("### **Aapke Data ke INR Price ko Predict Karega**")
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/48/car.png")
        st.title("Navigation")
        page = st.radio("Go to", ["Home", "Price Prediction", "EMI Calculator", "About"])
        
        if st.button("🔄 Retrain Model"):
            for cache in [st.cache_data, st.cache_resource]:
                cache.clear()
            st.session_state.model_trained = False
            st.session_state.df_clean = pd.DataFrame()
            st.rerun()
    
    # File upload - YAHAN SE DATA AAYEGA
    uploaded_file = st.file_uploader("📁 Apna CSV File Upload Karein", type=["csv"])
    
    # Load data
    if uploaded_file is not None:
        try:
            df_clean = load_data(uploaded_file)
            st.session_state.df_clean = df_clean
            
            # Show data preview
            with st.expander("👀 Data Preview"):
                st.dataframe(df_clean.head())
                st.write(f"**Data Shape:** {df_clean.shape}")
                
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            st.session_state.df_clean = pd.DataFrame()
    else:
        st.info("📝 Please upload your CSV file with car prices")
        st.session_state.df_clean = pd.DataFrame()
    
    # Train model if data available
    df_clean = st.session_state.df_clean
    
    if not df_clean.empty and 'Market_Price_INR' in df_clean.columns:
        if not st.session_state.model_trained:
            with st.spinner('🤖 Training AI model...'):
                try:
                    model_data = train_model(df_clean)
                    st.session_state.model = model_data
                    st.session_state.model_trained = True
                    st.session_state.model_ok = model_data['r2'] >= 0.70  # Reasonable threshold
                    
                    if st.session_state.model_ok:
                        st.success(f"✅ INR Price Prediction Model Ready!")
                    else:
                        st.warning("⚠ Model accuracy limited - using fallback methods")
                        
                except Exception as e:
                    st.error(f"❌ Model training failed: {e}")
                    st.session_state.model_ok = False
    else:
        st.session_state.model_ok = False
    
    # Page routing
    if page == "Home":
        st.subheader("🏠 Market Overview")
        
        if not df_clean.empty:
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_price = df_clean['Market_Price_INR'].mean()
                st.metric("Average Price", f"₹{avg_price:,.0f}")
            
            with col2:
                min_price = df_clean['Market_Price_INR'].min()
                st.metric("Minimum Price", f"₹{min_price:,.0f}")
            
            with col3:
                max_price = df_clean['Market_Price_INR'].max()
                st.metric("Maximum Price", f"₹{max_price:,.0f}")
            
            with col4:
                total_cars = len(df_clean)
                st.metric("Total Cars", f"{total_cars:,}")
            
            # Price distribution
            st.subheader("💰 Price Distribution")
            fig = px.histogram(df_clean, x='Market_Price_INR', 
                              title="Car Price Distribution in Your Data",
                              color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Brand analysis
            if 'Brand' in df_clean.columns:
                st.subheader("🏷️ Brands in Your Data")
                brand_stats = df_clean.groupby('Brand').agg({
                    'Market_Price_INR': ['count', 'mean', 'min', 'max']
                }).round(0)
                brand_stats.columns = ['Count', 'Avg Price', 'Min Price', 'Max Price']
                st.dataframe(brand_stats.sort_values('Count', ascending=False))
        
        else:
            st.info("📊 Upload a CSV file to see market insights")
    
    elif page == "Price Prediction":
        st.subheader("💰 Car Price Prediction")
        
        df_clean = st.session_state.df_clean
        
        if df_clean.empty:
            st.warning("❌ Please upload CSV file first for predictions")
            st.stop()
        
        # Input section
        st.markdown("### 🚗 Car Details Enter Karein")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Brand selection from uploaded data
            if 'Brand' in df_clean.columns:
                available_brands = sorted(df_clean['Brand'].unique().tolist())
                brand = st.selectbox("Brand", available_brands)
            else:
                brand = st.selectbox("Brand", list(CAR_CATALOG.keys()))
            
            # Model selection
            available_models = []
            if 'Model' in df_clean.columns and brand in df_clean['Brand'].values:
                available_models = sorted(df_clean[df_clean['Brand'] == brand]['Model'].unique().tolist())
            
            if not available_models and brand in CAR_CATALOG:
                available_models = CAR_CATALOG[brand]
            
            model_name = st.selectbox("Model", available_models or ["Select brand first"])
            
            if 'Year' in df_clean.columns:
                current_year = datetime.now().year
                year = st.number_input("Manufacturing Year", 
                                     min_value=1980, 
                                     max_value=current_year + 1, 
                                     value=current_year - 3)
            else:
                year = st.number_input("Manufacturing Year", value=2020)
        
        with col2:
            if 'Mileage' in df_clean.columns:
                mileage = st.number_input("Mileage (km)", 
                                        min_value=0, 
                                        max_value=500000, 
                                        value=30000)
            else:
                mileage = st.number_input("Mileage (km)", value=30000)
            
            if 'Fuel_Type' in df_clean.columns:
                fuel_options = sorted(df_clean['Fuel_Type'].unique().tolist())
                fuel = st.selectbox("Fuel Type", fuel_options)
            else:
                fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
            
            if 'Transmission' in df_clean.columns:
                transmission_options = sorted(df_clean['Transmission'].unique().tolist())
                transmission = st.selectbox("Transmission", transmission_options)
            else:
                transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            
            city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune"])
        
        if st.button("🎯 Predict INR Price", type="primary", use_container_width=True):
            with st.spinner("🔍 Calculating price..."):
                # Prepare input data
                input_data = {
                    'Brand': brand, 
                    'Model': model_name, 
                    'Year': year,
                    'Mileage': mileage,
                    'Fuel_Type': fuel,
                    'Transmission': transmission,
                    'City': city
                }
                
                final_price = None
                source = ""
                
                # Try AI model first
                if st.session_state.model_ok and st.session_state.model_trained:
                    predicted_price, pred_source = predict_inr_price(
                        st.session_state.model, input_data, df_clean
                    )
                    
                    if predicted_price is not None:
                        final_price = predicted_price
                        source = pred_source
                
                # Fallback methods
                if final_price is None:
                    web_price = get_web_price(brand, model_name, year, city)
                    if web_price:
                        final_price = web_price
                        source = "Web Data"
                    else:
                        # Simple estimation based on data
                        if not df_clean.empty:
                            brand_avg = df_clean['Market_Price_INR'].mean()
                            current_year = datetime.now().year
                            age = current_year - year
                            final_price = brand_avg * (1 - 0.10 * min(age, 10))
                            source = "Data Estimate"
                        else:
                            final_price = 500000  # Default fallback
                            source = "Default Estimate"
                
                # Display results
                st.success(f"✅ Price Predicted Using: **{source}**")
                
                # Calculate price range
                min_price = final_price * 0.90
                max_price = final_price * 1.10
                
                # Results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Minimum Price", f"₹{min_price:,.0f}")
                
                with col2:
                    st.metric("Fair Market Price", f"₹{final_price:,.0f}")
                
                with col3:
                    st.metric("Maximum Price", f"₹{max_price:,.0f}")
                
                # Visual representation
                st.subheader("📊 Price Range Analysis")
                
                fig = go.Figure()
                
                fig.add_trace(go.Indicator(
                    mode = "number+gauge+delta",
                    value = final_price,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    delta = {'reference': min_price},
                    number = {'prefix': "₹", 'font': {'size': 20}},
                    gauge = {
                        'shape': "bullet",
                        'axis': {'range': [None, max_price]},
                        'threshold': {
                            'line': {'color': "red", 'width': 2},
                            'thickness': 0.75,
                            'value': final_price},
                        'steps': [
                            {'range': [0, min_price], 'color': "lightgray"},
                            {'range': [min_price, final_price], 'color': "gray"}],
                        'bar': {'color': "darkblue"}}))
                
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                # Save prediction to history
                st.session_state.predictions.append({
                    'Brand': brand,
                    'Model': model_name, 
                    'Predicted_Price': f"₹{final_price:,.0f}",
                    'Price_Range': f"₹{min_price:,.0f} - ₹{max_price:,.0f}",
                    'Source': source,
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                
                st.balloons()
    
    elif page == "EMI Calculator":
        st.subheader("🧮 EMI Calculator")
        
        # EMI calculation logic (same as before)
        price = st.number_input("Car Price (₹)", 100000, 50000000, 1000000, 50000)
        down = st.slider("Down Payment (%)", 0, 50, 20)
        rate = st.slider("Interest Rate (%)", 5.0, 15.0, 9.5, 0.1)
        tenure = st.slider("Loan Tenure (years)", 1, 7, 5)
        
        # EMI calculation
        loan = price * (1 - down/100)
        r = rate / (12 * 100)
        months = tenure * 12
        emi = loan * r * ((1 + r)**months) / (((1 + r)**months) - 1) if loan > 0 else 0
        total = emi * months
        interest = total - loan
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Loan Amount", f"₹{loan:,.0f}")
            st.metric("Monthly EMI", f"₹{emi:,.0f}")
            st.metric("Total Interest", f"₹{interest:,.0f}")
            st.metric("Total Payment", f"₹{total:,.0f}")
        
        with col2:
            fig = go.Figure(data=[go.Pie(
                labels=['Principal', 'Interest'], 
                values=[loan, interest],
                hole=0.4, 
                marker_colors=['#4ecdc4', '#ff6b6b']
            )])
            fig.update_layout(title="EMI Breakdown")
            st.plotly_chart(fig, use_container_width=True)
    
    # Prediction History
    st.markdown("---")
    if st.session_state.predictions:
        with st.expander("📈 Prediction History"):
            hist_df = pd.DataFrame(st.session_state.predictions)
            st.dataframe(hist_df, use_container_width=True)
            
            if st.button("Clear History"):
                st.session_state.predictions = []
                st.rerun()
    
    st.markdown("### Made with ❤️ | **Aapke Data ke Hisaab se Price Predict Karega** 🚀")

# Run the application
if __name__ == "__main__":
    main()
