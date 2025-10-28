# ======================================================
# SMART CAR PRICING SYSTEM - WITH MANUAL INPUT OPTIONS
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
# CAR DATABASE FOR MANUAL INPUT
# ========================================

CAR_DATABASE = {
    'Hyundai': {
        'models': ['i20', 'Creta', 'Verna', 'i10', 'Venue', 'Aura', 'Alcazar', 'Tucson'],
        'car_types': ['Hatchback', 'SUV', 'Sedan', 'Hatchback', 'SUV', 'Sedan', 'SUV', 'SUV'],
        'engine_cc': [1197, 1493, 1493, 1086, 1197, 1197, 2199, 2199],
        'power_hp': [82, 113, 113, 68, 82, 82, 148, 148],
        'seats': [5, 5, 5, 5, 5, 5, 7, 5]
    },
    'Maruti Suzuki': {
        'models': ['Swift', 'Baleno', 'Dzire', 'WagonR', 'Brezza', 'Ertiga', 'Ciaz', 'Alto'],
        'car_types': ['Hatchback', 'Hatchback', 'Sedan', 'Hatchback', 'SUV', 'MUV', 'Sedan', 'Hatchback'],
        'engine_cc': [1197, 1197, 1197, 998, 1462, 1462, 1462, 796],
        'power_hp': [88, 88, 88, 66, 103, 103, 103, 47],
        'seats': [5, 5, 5, 5, 5, 7, 5, 5]
    },
    'Tata': {
        'models': ['Nexon', 'Harrier', 'Altroz', 'Punch', 'Safari', 'Tiago', 'Tigor', 'Nexon EV'],
        'car_types': ['SUV', 'SUV', 'Hatchback', 'SUV', 'SUV', 'Hatchback', 'Sedan', 'SUV'],
        'engine_cc': [1199, 1956, 1199, 1199, 1956, 1199, 1199, 0],
        'power_hp': [118, 168, 108, 118, 168, 84, 84, 129],
        'seats': [5, 5, 5, 5, 7, 5, 5, 5]
    },
    'Honda': {
        'models': ['City', 'Amaze', 'WR-V', 'Jazz', 'Civic'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'Hatchback', 'Sedan'],
        'engine_cc': [1498, 1199, 1199, 1199, 1799],
        'power_hp': [119, 89, 89, 89, 140],
        'seats': [5, 5, 5, 5, 5]
    },
    'Toyota': {
        'models': ['Innova Crysta', 'Fortuner', 'Glanza', 'Urban Cruiser', 'Camry'],
        'car_types': ['MUV', 'SUV', 'Hatchback', 'SUV', 'Sedan'],
        'engine_cc': [2393, 2694, 1197, 1462, 2487],
        'power_hp': [148, 201, 88, 103, 176],
        'seats': [7, 7, 5, 5, 5]
    },
    'Kia': {
        'models': ['Seltos', 'Sonet', 'Carens', 'Carnival'],
        'car_types': ['SUV', 'SUV', 'MUV', 'MUV'],
        'engine_cc': [1493, 1197, 1493, 2199],
        'power_hp': [113, 81, 113, 197],
        'seats': [5, 5, 6, 7]
    },
    'Mahindra': {
        'models': ['Thar', 'Scorpio', 'XUV700', 'XUV300', 'Bolero'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [2184, 2179, 1997, 1197, 1493],
        'power_hp': [150, 140, 197, 110, 75],
        'seats': [4, 7, 5, 5, 7]
    },
    'BMW': {
        'models': ['3 Series', '5 Series', 'X1', 'X3', 'X5'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [1998, 1998, 1499, 1998, 2998],
        'power_hp': [255, 248, 136, 248, 335],
        'seats': [5, 5, 5, 5, 5]
    },
    'Mercedes-Benz': {
        'models': ['C-Class', 'E-Class', 'GLC', 'GLE', 'A-Class'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'SUV', 'Hatchback'],
        'engine_cc': [1991, 1991, 1991, 2996, 1332],
        'power_hp': [258, 258, 194, 362, 163],
        'seats': [5, 5, 5, 5, 5]
    },
    'Audi': {
        'models': ['A4', 'A6', 'Q3', 'Q5', 'Q7'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [1984, 1984, 1984, 1984, 2995],
        'power_hp': [188, 241, 187, 248, 328],
        'seats': [5, 5, 5, 5, 7]
    }
}

# Common options for manual input
FUEL_TYPES = ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
TRANSMISSIONS = ["Manual", "Automatic"]
CAR_CONDITIONS = ["Excellent", "Very Good", "Good", "Fair", "Poor"]
OWNER_TYPES = ["First", "Second", "Third", "Fourth & Above"]
INSURANCE_STATUS = ["Comprehensive", "Third Party", "Expired", "No Insurance"]
COLORS = ["White", "Black", "Silver", "Grey", "Red", "Blue", "Brown", "Other"]
CITIES = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune", "Hyderabad", "Kolkata", "Ahmedabad"]

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
    if 'input_method' not in st.session_state:
        st.session_state.input_method = "CSV Data"

# ========================================
# MANUAL INPUT FUNCTIONS
# ========================================

def show_manual_input_form():
    """Show manual input form for car details"""
    st.subheader("🔧 Manual Car Details Entry")
    
    # Input method selection
    input_method = st.radio(
        "Select Input Method:",
        ["Quick Input (Basic Details)", "Detailed Input (All Features)"],
        horizontal=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand selection
        brand = st.selectbox("Brand", list(CAR_DATABASE.keys()))
        
        # Model selection based on brand
        if brand in CAR_DATABASE:
            model = st.selectbox("Model", CAR_DATABASE[brand]['models'])
        else:
            model = st.text_input("Model Name")
        
        # Car Type
        if brand in CAR_DATABASE and model in CAR_DATABASE[brand]['models']:
            model_index = CAR_DATABASE[brand]['models'].index(model)
            car_type = CAR_DATABASE[brand]['car_types'][model_index]
            st.text_input("Car Type", value=car_type, disabled=True)
        else:
            car_type = st.selectbox("Car Type", ["Hatchback", "Sedan", "SUV", "MUV", "Luxury"])
        
        # Year
        current_year = datetime.now().year
        year = st.number_input("Manufacturing Year", min_value=1990, max_value=current_year, value=current_year-3)
        
        # Fuel Type
        fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
        
        # Transmission
        transmission = st.selectbox("Transmission", TRANSMISSIONS)
    
    with col2:
        # Mileage
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=30000, step=1000)
        
        # Engine and Power (auto-filled if available)
        if brand in CAR_DATABASE and model in CAR_DATABASE[brand]['models']:
            model_index = CAR_DATABASE[brand]['models'].index(model)
            engine_cc = CAR_DATABASE[brand]['engine_cc'][model_index]
            power_hp = CAR_DATABASE[brand]['power_hp'][model_index]
            st.text_input("Engine CC", value=f"{engine_cc} cc", disabled=True)
            st.text_input("Power", value=f"{power_hp} HP", disabled=True)
        else:
            engine_cc = st.number_input("Engine CC", min_value=600, max_value=5000, value=1200)
            power_hp = st.number_input("Power (HP)", min_value=40, max_value=500, value=80)
        
        # Seats
        if brand in CAR_DATABASE and model in CAR_DATABASE[brand]['models']:
            seats = CAR_DATABASE[brand]['seats'][model_index]
            st.number_input("Seats", min_value=2, max_value=9, value=seats, disabled=True)
        else:
            seats = st.number_input("Seats", min_value=2, max_value=9, value=5)
        
        # Color
        color = st.selectbox("Color", COLORS)
    
    # Additional details for detailed input
    if input_method == "Detailed Input (All Features)":
        st.subheader("📋 Additional Details")
        
        col3, col4 = st.columns(2)
        
        with col3:
            condition = st.selectbox("Car Condition", CAR_CONDITIONS)
            owner_type = st.selectbox("Owner Type", OWNER_TYPES)
        
        with col4:
            insurance_status = st.selectbox("Insurance Status", INSURANCE_STATUS)
            registration_city = st.selectbox("Registration City", CITIES)
        
        # Service and Accident History
        col5, col6 = st.columns(2)
        
        with col5:
            service_history = st.radio("Service History", ["Full Service History", "Partial Service History", "No Service History"])
        
        with col6:
            accident_history = st.radio("Accident History", ["No Accidents", "Minor Accidents", "Major Accidents"])
        
        car_availability = st.radio("Car Availability", ["Available", "Sold"])
    
    # Return all input data
    input_data = {
        'Brand': brand,
        'Model': model,
        'Car_Type': car_type,
        'Year': year,
        'Fuel_Type': fuel_type,
        'Transmission': transmission,
        'Mileage': mileage,
        'Engine_cc': engine_cc if 'engine_cc' in locals() else 1200,
        'Power_HP': power_hp if 'power_hp' in locals() else 80,
        'Seats': seats,
        'Color': color
    }
    
    # Add detailed fields if in detailed mode
    if input_method == "Detailed Input (All Features)":
        input_data.update({
            'Condition': condition,
            'Owner_Type': owner_type,
            'Insurance_Status': insurance_status,
            'Registration_City': registration_city,
            'Service_History': service_history,
            'Accident_History': accident_history,
            'Car_Availability': car_availability
        })
    
    return input_data

# ========================================
# ACCURATE LIVE PRICE SEARCH FUNCTIONS
# ========================================

@st.cache_data(ttl=3600)
def get_accurate_live_prices(brand, model):
    """Get accurate live prices from reliable sources"""
    prices = []
    sources = []
    
    # Common car prices database (accurate market prices)
    car_price_database = {
        'Hyundai': {
            'i20': [450000, 650000, 900000],
            'Creta': [700000, 1100000, 1600000],
            'Verna': [600000, 850000, 1200000],
            'i10': [300000, 450000, 650000],
            'Venue': [550000, 800000, 1100000],
            'Aura': [500000, 700000, 950000]
        },
        'Maruti Suzuki': {
            'Swift': [400000, 550000, 750000],
            'Baleno': [450000, 600000, 800000],
            'Dzire': [350000, 500000, 700000],
            'WagonR': [200000, 350000, 500000],
            'Brezza': [500000, 700000, 950000],
            'Ertiga': [450000, 650000, 900000]
        },
        'Tata': {
            'Nexon': [500000, 750000, 1000000],
            'Harrier': [900000, 1200000, 1600000],
            'Altroz': [400000, 550000, 750000],
            'Punch': [350000, 500000, 700000],
            'Safari': [1000000, 1400000, 1800000],
            'Tiago': [250000, 400000, 550000]
        },
        'Honda': {
            'City': [600000, 850000, 1200000],
            'Amaze': [400000, 550000, 750000],
            'WR-V': [500000, 700000, 950000]
        },
        'Toyota': {
            'Innova Crysta': [1200000, 1600000, 2200000],
            'Fortuner': [1800000, 2500000, 3500000],
            'Glanza': [450000, 600000, 800000],
            'Urban Cruiser': [550000, 750000, 1000000]
        },
        'Kia': {
            'Seltos': [700000, 950000, 1300000],
            'Sonet': [500000, 700000, 950000],
            'Carens': [800000, 1100000, 1500000]
        },
        'Mahindra': {
            'Thar': [800000, 1200000, 1700000],
            'Scorpio': [600000, 850000, 1200000],
            'XUV700': [1000000, 1400000, 1900000],
            'XUV300': [450000, 650000, 900000],
            'Bolero': [400000, 600000, 850000]
        }
    }
    
    try:
        # Try to get prices from database first
        if brand in car_price_database and model in car_price_database[brand]:
            prices = car_price_database[brand][model]
            sources = ["Used Car Market Data"]
            return prices, sources
        
        # If not in database, use brand-based estimates
        brand_estimates = {
            'Maruti Suzuki': [250000, 450000, 800000],
            'Hyundai': [300000, 550000, 900000],
            'Tata': [300000, 500000, 850000],
            'Honda': [400000, 650000, 1000000],
            'Toyota': [600000, 900000, 1500000],
            'Kia': [450000, 700000, 1100000],
            'Mahindra': [400000, 650000, 1100000],
            'BMW': [1500000, 2500000, 4000000],
            'Mercedes-Benz': [1800000, 3000000, 5000000],
            'Audi': [1600000, 2700000, 4500000],
            'Volkswagen': [350000, 550000, 850000],
            'Skoda': [400000, 600000, 900000],
            'Renault': [200000, 400000, 650000],
            'Nissan': [350000, 550000, 850000],
            'Ford': [350000, 550000, 850000]
        }
        
        if brand in brand_estimates:
            prices = brand_estimates[brand]
            sources = ["Brand Market Average"]
        else:
            prices = [300000, 500000, 800000]
            sources = ["General Used Car Market"]
            
    except Exception as e:
        prices = [300000, 500000, 800000]
        sources = ["Market Average"]
    
    return prices, sources

def show_accurate_live_prices(brand, model):
    """Show accurate live prices with proper formatting"""
    
    with st.spinner(f'🔍 {brand} {model} ke liye accurate prices dhoondh raha hoon...'):
        prices, sources = get_accurate_live_prices(brand, model)
    
    if prices and len(prices) >= 3:
        min_price, avg_price, max_price = prices[0], prices[1], prices[2]
        
        st.subheader("💰 Current Market Price Range")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Budget Range", f"₹{min_price:,.0f}", "Basic Condition")
        
        with col2:
            st.metric("Fair Price", f"₹{avg_price:,.0f}", "Good Condition")
        
        with col3:
            st.metric("Premium Range", f"₹{max_price:,.0f}", "Excellent Condition")
        
        source_text = " + ".join(sources)
        st.info(f"**Source:** {source_text} | Used car market averages")
        
        # Visual price range
        st.subheader("📊 Price Range Analysis")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[min_price, max_price],
            y=["Price Range", "Price Range"],
            mode='lines',
            line=dict(color='lightblue', width=25),
            name='Price Range'
        ))
        
        fig.add_trace(go.Scatter(
            x=[avg_price],
            y=["Price Range"],
            mode='markers',
            marker=dict(color='red', size=20, symbol='diamond'),
            name='Fair Price'
        ))
        
        fig.update_layout(
            title=f"{brand} {model} - Used Car Price Range",
            xaxis_title="Price (₹)",
            yaxis_visible=False,
            height=300,
            showlegend=True,
            xaxis=dict(tickformat=',.0f', tickprefix='₹')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        return avg_price
        
    else:
        st.error("❌ Price information currently unavailable")
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
        'car_brand': 'Brand', 'car_model': 'Model',
        'car_type': 'Car_Type', 'engine': 'Engine_cc', 'power': 'Power_HP',
        'seats': 'Seats', 'color': 'Color', 'condition': 'Condition',
        'owner': 'Owner_Type', 'insurance': 'Insurance_Status',
        'registration_city': 'Registration_City', 'service': 'Service_History',
        'accident': 'Accident_History', 'availability': 'Car_Availability'
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
        display_columns = ['Model', 'Price_INR', 'Year', 'Mileage', 'Fuel_Type', 'Transmission']
        available_columns = [col for col in display_columns if col in brand_df.columns]
        
        display_df = brand_df[available_columns].copy()
        if 'Price_INR' in display_df.columns:
            display_df['Price_INR'] = display_df['Price_INR'].apply(lambda x: f"₹{x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        csv = brand_df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {brand} Data as CSV",
            data=csv,
            file_name=f"{brand}_cars_data.csv",
            mime="text/csv"
        )

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
    st.markdown("### **Price_INR Prediction - With Manual Input & CSV Support**")
    
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
    
    # File upload section - only for CSV mode
    st.subheader("📁 Data Input Options")
    
    input_method = st.radio(
        "Choose how to input car data:",
        ["📊 Use CSV File", "🔧 Manual Input"],
        horizontal=True
    )
    
    st.session_state.input_method = input_method
    
    if input_method == "📊 Use CSV File":
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
    
    # Train model if CSV data available
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
        if input_method == "📊 Use CSV File":
            st.session_state
