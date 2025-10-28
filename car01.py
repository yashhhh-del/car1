# ======================================================
# SMART CAR PRICING SYSTEM - WITH BRAND DATA DISPLAY
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
        # Try to find similar price column
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
    
    # Rename to Price_INR for consistency
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
    
    # Remove rows with missing Price_INR
    df = df.dropna(subset=['Price_INR'])
    st.info(f"✅ Removed rows with missing Price_INR: {original_rows} → {len(df)}")
    
    # Convert Price_INR to numeric
    df['Price_INR'] = pd.to_numeric(df['Price_INR'], errors='coerce')
    df = df.dropna(subset=['Price_INR'])
    st.info(f"✅ Cleaned numeric Price_INR: {len(df)} rows remaining")
    
    # Clean other columns
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
        
        # Store models for each brand
        st.session_state.available_models = {}
        st.session_state.brand_data = {}
        
        for brand in st.session_state.available_brands:
            brand_df = df[df['Brand'] == brand]
            models = sorted(brand_df['Model'].astype(str).unique().tolist())
            st.session_state.available_models[brand] = models
            
            # Store complete brand data for display
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
    
    # Key statistics
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
    
    # Show data table
    with st.expander(f"👀 View {brand} Cars Data", expanded=False):
        # Select important columns to display
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
        
        # Download button for brand data
        csv = brand_df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {brand} Data as CSV",
            data=csv,
            file_name=f"{brand}_cars_data.csv",
            mime="text/csv"
        )
    
    # Visualizations for the brand
    st.subheader(f"📈 {brand} Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution for the brand
        fig1 = px.histogram(brand_df, x='Price_INR', 
                           title=f"{brand} - Price Distribution",
                           color_discrete_sequence=['#FF6B6B'])
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Models price comparison
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
    
    # Year-wise analysis if available
    if 'Year' in brand_df.columns:
        st.subheader(f"📅 {brand} - Year-wise Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year_count = brand_df['Year'].value_counts().sort_index()
            fig3 = px.line(x=year_count.index, y=year_count.values,
                          title=f"{brand} - Cars by Manufacturing Year",
                          labels={'x': 'Year', 'y': 'Number of Cars'})
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            year_prices = brand_df.groupby('Year')['Price_INR'].mean()
            fig4 = px.line(x=year_prices.index, y=year_prices.values,
                          title=f"{brand} - Average Price by Year",
                          labels={'x': 'Year', 'y': 'Average Price_INR'})
            st.plotly_chart(fig4, use_container_width=True)

# ========================================
# MODEL TRAINING FUNCTIONS
# ========================================

@st.cache_resource
def train_model(df):
    """Train model to predict Price_INR from your data"""
    current_year = datetime.now().year
    df_model = df.copy()
    
    st.write("🔧 Preparing features for Price_INR prediction...")
    
    # FEATURE ENGINEERING
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
    
    # ENCODE CATEGORICAL VARIABLES
    cat_cols = df_model.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le

    # PREPARE FEATURES AND TARGET (Price_INR)
    X = df_model.drop(columns=['Price_INR'], errors='ignore')
    y = df_model['Price_INR']  # TARGET: Price_INR
    
    st.write(f"🎯 **Target Variable:** Price_INR")
    st.write(f"📊 **Features used:** {len(X.columns)} columns")
    
    # Scale features
    X_scaled = StandardScaler().fit_transform(X)
    
    # HYPERPARAMETER TUNING
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
    
    with st.spinner('🤖 Training model to predict Price_INR...'):
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
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Apply same encoding as training
        for col in model_data['encoders']:
            if col in input_df.columns:
                try: 
                    input_df[col] = model_data['encoders'][col].transform([input_data[col]])[0]
                except:
                    # If new category, use default value
                    input_df[col] = 0
        
        # Ensure all features are present
        for col in model_data['features']:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select and order features correctly
        input_df = input_df[model_data['features']]
        
        # Scale features
        input_scaled = model_data['scaler'].transform(input_df)
        
        # Predict Price_INR
        predicted_price = model_data['model'].predict(input_scaled)[0]
        
        return predicted_price, "AI Model (Your Data)"
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return None, "Error"

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    # Initialize session state
    initialize_session_state()
    
    # Page config
    st.set_page_config(
        page_title="Car Price Predictor", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    st.title("🚗 Car Price Prediction System")
    st.markdown("### **Price_INR Prediction - With Brand Data Display**")
    
    # Sidebar navigation
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
    
    # File upload section
    st.subheader("📁 Apna CSV File Upload Karein")
    uploaded_file = st.file_uploader("Choose CSV file with Price_INR column", type=["csv"])
    
    # Load data
    if uploaded_file is not None:
        try:
            df_clean = load_data(uploaded_file)
            st.session_state.df_clean = df_clean
            
            # Show data preview
            with st.expander("👀 Complete Data Preview", expanded=False):
                st.dataframe(df_clean.head(10))
                st.write(f"**Dataset Shape:** {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
                
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            st.session_state.df_clean = pd.DataFrame()
    else:
        st.info("📝 Please upload your CSV file with Price_INR column")
        st.session_state.df_clean = pd.DataFrame()
    
    # Train model if data available
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
    
    # Page routing
    if page == "Data Overview":
        st.subheader("📊 Your Data Overview")
        
        if not df_clean.empty:
            # Overall statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cars", len(df_clean))
            with col2:
                st.metric("Avg Price_INR", f"₹{df_clean['Price_INR'].mean():,.0f}")
            with col3:
                st.metric("Brands", len(st.session_state.available_brands))
            with col4:
                st.metric("Price Range", f"₹{df_clean['Price_INR'].min():,.0f} - ₹{df_clean['Price_INR'].max():,.0f}")
            
            # Price distribution
            st.subheader("💰 Price_INR Distribution")
            fig1 = px.histogram(df_clean, x='Price_INR', 
                               title="Distribution of Price_INR in Your Data",
                               color_discrete_sequence=['#FF6B6B'])
            st.plotly_chart(fig1, use_container_width=True)
            
            # Brand analysis
            if 'Brand' in df_clean.columns:
                st.subheader("🏷️ Brands Overview")
                
                # Brand count
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
        
        # Input section - ONLY FROM USER'S DATA
        st.markdown("### 🚗 Car Details (From Your Data)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Brand selection - ONLY from uploaded data
            if st.session_state.available_brands:
                brand = st.selectbox("Select Brand", st.session_state.available_brands)
                st.info(f"📊 {len(st.session_state.available_models.get(brand, []))} models available for {brand}")
            else:
                st.error("❌ No Brand column found in your data")
                return
            
            # Model selection - ONLY from selected brand in data
            if brand in st.session_state.available_models:
                available_models = st.session_state.available_models[brand]
                if available_models:
                    model_name = st.selectbox("Select Model", available_models)
                else:
                    st.error(f"❌ No models found for brand '{brand}' in your data")
                    return
            else:
                st.error(f"❌ Brand '{brand}' not found in available models")
                return
            
            # Year input
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
            # Mileage input
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
            
            # Fuel Type - FROM DATA
            if 'Fuel_Type' in df_clean.columns:
                fuel_options = sorted(df_clean['Fuel_Type'].astype(str).unique().tolist())
                fuel = st.selectbox("Fuel Type", fuel_options)
            else:
                fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
            
            # Transmission - FROM DATA
            if 'Transmission' in df_clean.columns:
                transmission_options = sorted(df_clean['Transmission'].astype(str).unique().tolist())
                transmission = st.selectbox("Transmission", transmission_options)
            else:
                transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            
            # City - FROM DATA if available
            if 'City' in df_clean.columns:
                city_options = sorted(df_clean['City'].astype(str).unique().tolist())
                city = st.selectbox("City", city_options)
            else:
                city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune"])
        
        # SHOW BRAND DATA WHEN BRAND IS SELECTED
        if brand:
            show_brand_data(brand)
        
        # Prediction button
        if st.button("🎯 Predict Price_INR", type="primary", use_container_width=True):
            with st.spinner("🔍 Predicting Price_INR..."):
                # Prepare input data
                input_data = {
                    'Brand': brand, 
                    'Model': model_name, 
                    'Year': year,
                    'Mileage': mileage,
                    'Fuel_Type': fuel,
                    'Transmission': transmission
                }
                
                # Add City if available in original data
                if 'City' in df_clean.columns:
                    input_data['City'] = city
                
                # Predict using AI model
                final_price, source = predict_price_inr(st.session_state.model, input_data, df_clean)
                
                if final_price is None:
                    st.error("❌ Prediction failed. Please try again.")
                    return
                
                # Display results
                st.success(f"✅ **Prediction Source:** {source}")
                
                # Calculate price range (10% variation)
                min_price = final_price * 0.90
                max_price = final_price * 1.10
                
                # Show results in metrics
                st.subheader("💰 Predicted Price_INR")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Minimum Expected", f"₹{min_price:,.0f}")
                
                with col2:
                    st.metric("Fair Market Price", f"₹{final_price:,.0f}")
                
                with col3:
                    st.metric("Maximum Expected", f"₹{max_price:,.0f}")
                
                # Compare with actual brand data
                if brand in st.session_state.brand_data:
                    brand_avg = st.session_state.brand_data[brand]['Price_INR'].mean()
                    st.info(f"📊 Compared to {brand}'s average in your data: ₹{brand_avg:,.0f}")
                
                st.balloons()
                
                # Save to prediction history
                prediction_record = {
                    'Brand': brand,
                    'Model': model_name, 
                    'Year': year,
                    'Predicted_Price_INR': f"₹{final_price:,.0f}",
                    'Price_Range': f"₹{min_price:,.0f} - ₹{max_price:,.0f}",
                    'Source': source,
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
    
    # Prediction History
    st.markdown("---")
    if st.session_state.predictions:
        with st.expander("📈 Prediction History", expanded=False):
            hist_df = pd.DataFrame(st.session_state.predictions)
            st.dataframe(hist_df, use_container_width=True)
            
            if st.button("Clear History"):
                st.session_state.predictions = []
                st.rerun()

# Run the application
if __name__ == "__main__":
    main()
