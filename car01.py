# ======================================================
# SMART CAR PRICING SYSTEM - COMPLETE MERGED VERSION
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
    if 'available_brands' not in st.session_state:
        st.session_state.available_brands = []
    if 'available_models' not in st.session_state:
        st.session_state.available_models = {}
    if 'brand_data' not in st.session_state:
        st.session_state.brand_data = {}

# ========================================
# IMPROVED LIVE PRICE SEARCH FUNCTIONS
# ========================================

@st.cache_data(ttl=3600)
def get_live_car_prices_improved(brand, model, year=None):
    """Get live car prices from multiple sources with better error handling"""
    prices = []
    sources = []
    
    # Format search query
    brand_clean = brand.replace(' ', '-').lower()
    model_clean = model.replace(' ', '-').lower()
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    websites = [
        {
            'name': 'CarDekho',
            'url': f'https://www.cardekho.com/used-{brand_clean}-{model_clean}-cars',
            'pattern': r'₹\s*([\d,.]+)\s*Lakh|\$?\s*([\d,]+)\s*'
        },
        {
            'name': 'CarWale', 
            'url': f'https://www.carwale.com/used/{brand_clean}-{model_clean}-cars/',
            'pattern': r'₹\s*([\d,.]+)\s*Lakh|price.*?₹\s*([\d,]+)'
        }
    ]
    
    for website in websites:
        try:
            response = requests.get(website['url'], headers=headers, timeout=8)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                price_patterns = [
                    r'₹\s*([\d,.]+)\s*Lakh',
                    r'₹\s*([\d,.]+)',
                    r'price.*?₹\s*([\d,]+)',
                    r'Rs\.\s*([\d,]+)'
                ]
                
                for pattern in price_patterns:
                    matches = re.findall(pattern, soup.get_text(), re.IGNORECASE)
                    for match in matches:
                        if match:
                            price_str = str(match).replace(',', '').replace('.', '')
                            try:
                                price = float(price_str)
                                if 'lakh' in pattern.lower():
                                    price = price * 100000
                                if 10000 <= price <= 50000000:
                                    prices.append(price)
                                    sources.append(website['name'])
                                    break
                            except ValueError:
                                continue
                        
        except Exception as e:
            continue
    
    if not prices:
        return get_estimated_prices(brand, model), ["Estimated"]
    
    return prices, sources

def get_estimated_prices(brand, model):
    """Provide estimated prices when web scraping fails"""
    brand_base_prices = {
        'Maruti Suzuki': 500000, 'Hyundai': 600000, 'Honda': 700000, 'Toyota': 800000,
        'Tata': 450000, 'Mahindra': 550000, 'Kia': 650000, 'BMW': 2500000,
        'Mercedes-Benz': 3000000, 'Audi': 2800000, 'Volkswagen': 600000,
        'Skoda': 650000, 'Ford': 550000, 'Renault': 500000, 'Nissan': 550000
    }
    
    base_price = brand_base_prices.get(brand, 600000)
    
    prices = [
        base_price * 0.8,   # Min
        base_price,         # Avg
        base_price * 1.2    # Max
    ]
    
    return prices

def show_live_prices_improved(brand, model):
    """Show live prices with better fallback options"""
    
    with st.spinner(f'🔍 Live prices dhoondh raha hoon {brand} {model} ke liye...'):
        prices, sources = get_live_car_prices_improved(brand, model)
    
    if prices and len(prices) > 0:
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
        
        st.subheader("🌐 Live Market Prices")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Minimum Price", f"₹{min_price:,.0f}")
        with col2:
            st.metric("Average Price", f"₹{avg_price:,.0f}")
        with col3:
            st.metric("Maximum Price", f"₹{max_price:,.0f}")
        
        source_text = ", ".join(set(sources))
        if "Estimated" in source_text:
            st.warning(f"**Sources:** {source_text} (Based on market averages)")
        else:
            st.success(f"**Sources:** {source_text}")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Min', 'Avg', 'Max'],
            y=[min_price, avg_price, max_price],
            marker_color=['#ff6b6b', '#1a936f', '#ffe66d'],
            text=[f"₹{min_price:,.0f}", f"₹{avg_price:,.0f}", f"₹{max_price:,.0f}"],
            textposition='outside'
        ))
        fig.update_layout(
            title=f"{brand} {model} - Price Range",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return avg_price
    else:
        st.error("❌ Live prices currently unavailable - using AI prediction only")
        return None

# ========================================
# DATA LOADING FUNCTIONS
# ========================================

@st.cache_data
def load_data(file):
    """Load and clean CSV data - SPECIFICALLY FOR Price_INR"""
    df = pd.read_csv(file)
    
    st.info(f"📁 Original columns: {list(df.columns)}")
    
    # FIND Price_INR COLUMN SPECIFICALLY
    if 'Price_INR' in df.columns:
        price_col = 'Price_INR'
        st.success("✅ Price_INR column found!")
    else:
        price_col = None
        price_keywords = ['price_inr', 'price', 'inr', 'amount', 'cost']
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in price_keywords):
                price_col = col
                st.success(f"✅ Price column found: {col} → renaming to Price_INR")
                break
        
        if not price_col:
            st.error("❌ Price_INR column not found in CSV!")
            st.info("Please make sure your CSV has 'Price_INR' column")
            return pd.DataFrame()
    
    if price_col != 'Price_INR':
        df = df.rename(columns={price_col: 'Price_INR'})
    
    # STANDARDIZE COLUMN NAMES
    rename_map = {
        'brand': 'Brand', 'model': 'Model', 'year': 'Year', 
        'mileage': 'Mileage', 'fuel': 'Fuel_Type', 
        'transmission': 'Transmission', 'city': 'City',
        'company': 'Brand', 'car_name': 'Model', 'kms_driven': 'Mileage',
        'car_brand': 'Brand', 'car_model': 'Model'
    }
    
    columns_renamed = []
    for old, new in rename_map.items():
        for col in df.columns:
            if old in col.lower() and col != new:
                df = df.rename(columns={col: new})
                columns_renamed.append(f"'{col}' → '{new}'")
                break
    
    if columns_renamed:
        st.info(f"🔄 Columns renamed: {', '.join(columns_renamed)}")
    
    # CLEAN DATA FOR Price_INR PREDICTION
    original_rows = len(df)
    
    df = df.dropna(subset=['Price_INR'])
    st.info(f"✅ Removed rows with missing Price_INR: {original_rows} → {len(df)}")
    
    df['Price_INR'] = pd.to_numeric(df['Price_INR'], errors='coerce')
    df = df.dropna(subset=['Price_INR'])
    st.info(f"✅ Cleaned numeric Price_INR: {len(df)} rows remaining")
    
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year'])
        df['Year'] = df['Year'].astype(int)
        st.info(f"✅ Cleaned Year column: {df['Year'].min()} - {df['Year'].max()}")
    
    if 'Mileage' in df.columns:
        df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
        df = df.dropna(subset=['Mileage'])
        st.info(f"✅ Cleaned Mileage column: {df['Mileage'].min():,} - {df['Mileage'].max():,} km")
    
    # Store available brands and models
    if 'Brand' in df.columns:
        st.session_state.available_brands = sorted(df['Brand'].astype(str).unique().tolist())
        st.info(f"✅ Found {len(st.session_state.available_brands)} brands in data")
        
        st.session_state.available_models = {}
        st.session_state.brand_data = {}
        
        for brand in st.session_state.available_brands:
            brand_df = df[df['Brand'] == brand]
            models = sorted(brand_df['Model'].astype(str).unique().tolist())
            st.session_state.available_models[brand] = models
            st.session_state.brand_data[brand] = brand_df
    
    st.success(f"🎯 Final dataset: {len(df)} cars, Price_INR range: ₹{df['Price_INR'].min():,} to ₹{df['Price_INR'].max():,}")
    
    return df

# ========================================
# BRAND DATA DISPLAY FUNCTION
# ========================================

def show_brand_data(brand):
    """Show actual data for the selected brand from CSV"""
    if brand not in st.session_state.brand_data:
        return
    
    brand_df = st.session_state.brand_data[brand]
    
    st.subheader(f"📊 Actual Data for {brand} from Your CSV")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cars = len(brand_df)
        st.metric("Total Cars", total_cars)
    
    with col2:
        avg_price = brand_df['Price_INR'].mean()
        st.metric("Avg Price", f"₹{avg_price:,.0f}")
    
    with col3:
        min_price = brand_df['Price_INR'].min()
        st.metric("Min Price", f"₹{min_price:,.0f}")
    
    with col4:
        max_price = brand_df['Price_INR'].max()
        st.metric("Max Price", f"₹{max_price:,.0f}")
    
    with st.expander(f"👀 View {brand} Cars Data", expanded=False):
        display_columns = ['Model', 'Price_INR']
        if 'Year' in brand_df.columns:
            display_columns.append('Year')
        if 'Mileage' in brand_df.columns:
            display_columns.append('Mileage')
        if 'Fuel_Type' in brand_df.columns:
            display_columns.append('Fuel_Type')
        if 'Transmission' in brand_df.columns:
            display_columns.append('Transmission')
        
        display_df = brand_df[display_columns].copy()
        display_df['Price_INR'] = display_df['Price_INR'].apply(lambda x: f"₹{x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        csv = brand_df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {brand} Data as CSV",
            data=csv,
            file_name=f"{brand}_cars_data.csv",
            mime="text/csv"
        )
    
    st.subheader(f"📈 {brand} Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(brand_df, x='Price_INR', 
                           title=f"{brand} - Price Distribution",
                           color_discrete_sequence=['#FF6B6B'])
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if len(brand_df['Model'].unique()) > 1:
            model_prices = brand_df.groupby('Model')['Price_INR'].mean().sort_values(ascending=False)
            fig2 = px.bar(x=model_prices.values, y=model_prices.index,
                         orientation='h',
                         title=f"{brand} - Models by Average Price",
                         labels={'x': 'Price_INR', 'y': 'Model'},
                         color_discrete_sequence=['#4ECDC4'])
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info(f"Only one model found for {brand}")

# ========================================
# MODEL TRAINING FUNCTIONS
# ========================================

@st.cache_resource
def train_model(df):
    """Train model to predict Price_INR from your data"""
    current_year = datetime.now().year
    df_model = df.copy()
    
    st.write("🔧 Preparing features for Price_INR prediction...")
    
    features_added = []
    
    if 'Year' in df_model.columns:
        df_model['Car_Age'] = current_year - df_model['Year']
        features_added.append('Car_Age')
    
    if 'Brand' in df_model.columns:
        brand_avg = df_model.groupby('Brand')['Price_INR'].mean()
        df_model['Brand_Avg_Price'] = df_model['Brand'].map(brand_avg)
        features_added.append('Brand_Avg_Price')
    
    if features_added:
        st.info(f"✅ Added features: {', '.join(features_added)}")
    
    cat_cols = df_model.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le

    X = df_model.drop(columns=['Price_INR'], errors='ignore')
    y = df_model['Price_INR']
    
    st.write(f"🎯 **Target Variable:** Price_INR")
    st.write(f"📊 **Features used:** {len(X.columns)} columns")
    
    X_scaled = StandardScaler().fit_transform(X)
    
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
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    with st.spinner('🤖 Training model to predict Price_INR...'):
        grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    y_pred = best_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='r2')
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    st.success(f"""
    🎯 **Price_INR Prediction Model Ready!**
    
    **Performance Metrics:**
    - R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)
    - Mean Absolute Error: ₹{mae:,.0f}
    - Mean Absolute % Error: {mape:.2f}%
    - RMSE: ₹{rmse:,.0f}
    - Cross-val Consistency: {cv_scores.mean()*100:.2f}%
    
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

def predict_price_inr(model_data, input_data, df_clean):
    """Predict Price_INR using trained model"""
    try:
        input_df = pd.DataFrame([input_data])
        
        for col in model_data['encoders']:
            if col in input_df.columns:
                try: 
                    input_df[col] = model_data['encoders'][col].transform([input_data[col]])[0]
                except:
                    input_df[col] = 0
        
        for col in model_data['features']:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[model_data['features']]
        input_scaled = model_data['scaler'].transform(input_df)
        predicted_price = model_data['model'].predict(input_scaled)[0]
        
        return predicted_price, "AI Model (Your Data)"
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return None, "Error"

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    initialize_session_state()
    
    st.set_page_config(
        page_title="Car Price Predictor", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    st.title("🚗 Car Price Prediction System")
    st.markdown("### **Price_INR Prediction - With Live Market Prices**")
    
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/48/car.png")
        st.title("Navigation")
        page = st.radio("Go to", ["Data Overview", "Price Prediction", "Brand Analysis"])
        
        if st.button("🔄 Retrain Model"):
            for cache in [st.cache_data, st.cache_resource]:
                cache.clear()
            st.session_state.model_trained = False
            st.session_state.df_clean = pd.DataFrame()
            st.rerun()
    
    st.subheader("📁 Apna CSV File Upload Karein")
    uploaded_file = st.file_uploader("Choose CSV file with Price_INR column", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_clean = load_data(uploaded_file)
            st.session_state.df_clean = df_clean
            
            with st.expander("👀 Complete Data Preview", expanded=False):
                st.dataframe(df_clean.head(10))
                st.write(f"**Dataset Shape:** {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
                
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            st.session_state.df_clean = pd.DataFrame()
    else:
        st.info("📝 Please upload your CSV file with Price_INR column")
        st.session_state.df_clean = pd.DataFrame()
    
    df_clean = st.session_state.df_clean
    
    if not df_clean.empty and 'Price_INR' in df_clean.columns:
        if not st.session_state.model_trained:
            with st.spinner('🤖 Training AI model on your Price_INR data...'):
                try:
                    model_data = train_model(df_clean)
                    st.session_state.model = model_data
                    st.session_state.model_trained = True
                    st.session_state.model_ok = model_data['r2'] >= 0.70
                    
                    if st.session_state.model_ok:
                        st.success("✅ Price_INR Prediction Model Ready!")
                    else:
                        st.warning("⚠ Model accuracy limited - consider adding more data")
                        
                except Exception as e:
                    st.error(f"❌ Model training failed: {e}")
                    st.session_state.model_ok = False
        else:
            st.success("✅ Model already trained and ready for predictions!")
    else:
        st.session_state.model_ok = False
    
    if page == "Data Overview":
        st.subheader("📊 Your Data Overview")
        
        if not df_clean.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cars", len(df_clean))
            with col2:
                st.metric("Avg Price_INR", f"₹{df_clean['Price_INR'].mean():,.0f}")
            with col3:
                st.metric("Brands", len(st.session_state.available_brands))
            with col4:
                st.metric("Price Range", f"₹{df_clean['Price_INR'].min():,.0f} - ₹{df_clean['Price_INR'].max():,.0f}")
            
            st.subheader("💰 Price_INR Distribution")
            fig1 = px.histogram(df_clean, x='Price_INR', 
                               title="Distribution of Price_INR in Your Data",
                               color_discrete_sequence=['#FF6B6B'])
            st.plotly_chart(fig1, use_container_width=True)
            
            if 'Brand' in df_clean.columns:
                st.subheader("🏷️ Brands Overview")
                brand_count = df_clean['Brand'].value_counts().head(15)
                fig2 = px.bar(x=brand_count.values, y=brand_count.index,
                             orientation='h',
                             title="Top 15 Brands by Count",
                             color_discrete_sequence=['#4ECDC4'])
                st.plotly_chart(fig2, use_container_width=True)
        
        else:
            st.info("📊 Upload a CSV file to see data insights")
    
    elif page == "Price Prediction":
        st.subheader("💰 Price_INR Prediction")
        
        df_clean = st.session_state.df_clean
        
        if df_clean.empty:
            st.warning("❌ Please upload CSV file first for predictions")
            return
        
        if not st.session_state.model_trained:
            st.warning("⏳ Model training in progress... Please wait")
            return
        
        st.success("🎯 Model ready! Enter car details below:")
        
        st.markdown("### 🚗 Car Details (From Your Data)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.available_brands:
                brand = st.selectbox("Select Brand", st.session_state.available_brands)
                st.info(f"📊 {len(st.session_state.available_models.get(brand, []))} models available for {brand}")
            else:
                st.error("❌ No Brand column found in your data")
                return
            
            if brand in st.session_state.available_models:
                available_models = st.session_state.available_models[brand]
                if available_models:
                    model_name = st.selectbox("Select Model", available_models)
                    
                    if brand and model_name:
                        live_avg_price = show_live_prices_improved(brand, model_name)
                        
                else:
                    st.error(f"❌ No models found for brand '{brand}' in your data")
                    return
            else:
                st.error(f"❌ Brand '{brand}' not found in available models")
                return
            
            if 'Year' in df_clean.columns:
                current_year = datetime.now().year
                year_data = df_clean[df_clean['Brand'] == brand]
                if not year_data.empty:
                    min_year = int(year_data['Year'].min())
                    max_year = int(year_data['Year'].max())
                    default_year = max(min_year, current_year - 3)
                    year = st.number_input("Manufacturing Year", 
                                         min_value=min_year, 
                                         max_value=max_year, 
                                         value=default_year)
                else:
                    year = st.number_input("Manufacturing Year", 
                                         min_value=1990, 
                                         max_value=current_year, 
                                         value=current_year - 3)
        
        with col2:
            if 'Mileage' in df_clean.columns:
                mileage_data = df_clean[df_clean['Brand'] == brand]
                if not mileage_data.empty:
                    avg_mileage = int(mileage_data['Mileage'].mean())
                    mileage = st.number_input("Mileage (km)", 
                                            min_value=0, 
                                            max_value=500000, 
                                            value=avg_mileage)
                else:
                    mileage = st.number_input("Mileage (km)", value=30000)
            
            if 'Fuel_Type' in df_clean.columns:
                fuel_options = sorted(df_clean['Fuel_Type'].astype(str).unique().tolist())
                fuel = st.selectbox("Fuel Type", fuel_options)
            else:
                fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
            
            if 'Transmission' in df_clean.columns:
                transmission_options = sorted(df_clean['Transmission'].astype(str).unique().tolist())
                transmission = st.selectbox("Transmission", transmission_options)
            else:
                transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            
            if 'City' in df_clean.columns:
                city_options = sorted(df_clean['City'].astype(str).unique().tolist())
                city = st.selectbox("City", city_options)
            else:
                city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune"])
        
        if brand:
            show_brand_data(brand)
        
        if st.button("🎯 Predict Price_INR", type="primary", use_container_width=True):
            with st.spinner("🔍 Predicting Price_INR..."):
                input_data = {
                    'Brand': brand, 
                    'Model': model_name, 
                    'Year': year,
                    'Mileage': mileage,
                    'Fuel_Type': fuel,
                    'Transmission': transmission
                }
                
                if 'City' in df_clean.columns:
                    input_data['City'] = city
                
                final_price, source = predict_price_inr(st.session_state.model, input_data, df_clean)
                
                if final_price is None:
                    st.error("❌ Prediction failed. Please try again.")
                    return
                
                st.success("📊 **Price Prediction Results**")
                
                if 'live_avg_price' in locals() and live_avg_price is not None:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("AI Prediction", f"₹{final_price:,.0f}", "Your Data Based")
                    
                    with col2:
                        st.metric("Market Reference", f"₹{live_avg_price:,.0f}", "Market Data")
                    
                    with col3:
                        recommended = (final_price + live_avg_price) / 2
                        st.metric("Recommended", f"₹{recommended:,.0f}", "Balanced")
                    
                    st.subheader("📈 Price Comparison")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['AI Prediction', 'Market Reference', 'Recommended'],
                        y=[final_price, live_avg_price, recommended],
                        marker_color=['#1a936f', '#ff6b6b', '#ffe66d'],
                        text=[f"₹{final_price:,.0f}", f"₹{live_avg_price:,.0f}", f"₹{recommended:,.0f}"],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title=f"{brand} {model_name} - Price Comparison",
                        height=500,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("AI Prediction", f"₹{final_price:,.0f}", "Your Data Based")
                    
                    with col2:
                        min_price = final_price * 0.85
                        max_price = final_price * 1.15
                        st.metric("Expected Range", f"₹{min_price:,.0f} - ₹{max_price:,.0f}")
                    
                    st.subheader("💰 Predicted Price")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Indicator(
                        mode = "number+delta",
                        value = final_price,
                        number = {'prefix': "₹", 'valueformat': ",.0f"},
                        delta = {'reference': final_price * 0.9, 'position': "bottom"},
                        title = {"text": f"{brand} {model_name}<br>Predicted Price"}
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("💡 Buying/Selling Advice")
                
                advice_col1, advice_col2 = st.columns(2)
                
                with advice_col1:
                    st.info("""
                    **✅ If Buying:**
                    - Compare with local dealers
                    - Check vehicle service history
                    - Get professional inspection
                    - Verify ownership documents
                    - Test drive thoroughly
                    """)
                
                with advice_col2:
                    st.info("""
                    **✅ If Selling:**
                    - Highlight maintenance records
                    - Clean the car completely
                    - Fix minor issues
                    - Provide clear photos
                    - Be realistic about pricing
                    """)
                
                st.balloons()
                
                prediction_record = {
                    'Brand': brand,
                    'Model': model_name, 
                    'Year': year,
                    'AI_Prediction': f"₹{final_price:,.0f}",
                    'Market_Reference': f"₹{live_avg_price:,.0f}" if 'live_avg_price' in locals() and live_avg_price else "N/A",
                    'Recommended': f"₹{recommended:,.0f}" if 'live_avg_price' in locals() and live_avg_price else f"₹{final_price:,.0f}",
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                
                st.session_state.predictions.append(prediction_record)
    
    elif page == "Brand Analysis":
        st.subheader("🏷️ Brand-wise Analysis")
        
        if not df_clean.empty and st.session_state.available_brands:
            selected_brand = st.selectbox("Select Brand for Detailed Analysis", 
                                         st.session_state.available_brands)
            
            if selected_brand:
                show_brand_data(selected_brand)
        else:
            st.info("📊 Upload a CSV file to see brand analysis")
    
    st.markdown("---")
    if st.session_state.predictions:
        with st.expander("📈 Prediction History", expanded=False):
            hist_df = pd.DataFrame(st.session_state.predictions)
            st.dataframe(hist_df, use_container_width=True)
            
            if st.button("Clear History"):
                st.session_state.predictions = []
                st.rerun()

if __name__ == "__main__":
    main()
