# ======================================================
# SMART CAR PRICING SYSTEM - BUSINESS EVALUATION READY
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Smart Car Pricing Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .big-metric {font-size: 24px; font-weight: bold; color: #1f77b4;}
    .section-header {background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                     padding: 10px; border-radius: 5px; color: white; font-weight: bold;}
    .success-box {background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 5px solid #28a745;}
    .info-box {background-color: #d1ecf1; padding: 15px; border-radius: 5px; border-left: 5px solid #17a2b8;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'evaluation_scores' not in st.session_state:
    st.session_state.evaluation_scores = {}

# Title
st.markdown("<h1 style='text-align: center; color: #667eea;'>🚗 Smart Car Pricing System - Business Edition</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Professional Used Car Valuation with AI-Powered Insights</h4>", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/car--v1.png", width=80)
    st.title("📊 Navigation")
    page = st.radio("", [
        "🏠 Dashboard",
        "💰 Price Prediction",
        "📊 Compare Cars",
        "📈 Market Insights",
        "🔧 Model Performance",
        "🧮 EMI Calculator",
        "📋 Business Report"
    ])
    
    st.markdown("---")
    st.markdown("### 🎯 System Status")
    if st.session_state.model_trained:
        st.success("✅ Model Active")
    else:
        st.warning("⏳ Upload Data")

# File Upload
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"], help="Upload your used car dataset")

if uploaded_file is None:
    st.info("👆 Please upload a CSV file to begin analysis")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📋 Required Columns:")
        st.code("""
- Brand (e.g., Maruti, Honda)
- Model (e.g., Swift, City)
- Year (e.g., 2020)
- Price/Market_Price (₹)
- Mileage (optional)
- Fuel_Type (optional)
- Transmission (optional)
        """)
    
    with col2:
        st.markdown("### 📄 Sample Format:")
        st.code("""Brand,Model,Year,Mileage,Fuel_Type,Transmission,Price
Maruti,Swift,2020,15000,Petrol,Manual,550000
Honda,City,2019,20000,Petrol,Automatic,900000
Hyundai,Creta,2021,10000,Diesel,Manual,1400000""")
    
    st.markdown("---")
    st.markdown("### 🎯 Business Value Proposition")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🎯 For Dealers**")
        st.write("• Competitive pricing strategy")
        st.write("• Inventory valuation")
        st.write("• Profit optimization")
    with col2:
        st.markdown("**👥 For Buyers**")
        st.write("• Fair price validation")
        st.write("• Negotiation insights")
        st.write("• Investment decisions")
    with col3:
        st.markdown("**💼 For Business**")
        st.write("• Market trend analysis")
        st.write("• Data-driven decisions")
        st.write("• Revenue forecasting")
    
    st.stop()

# Load and process data
@st.cache_data
def load_and_clean_data(file):
    df = pd.read_csv(file)
    
    # Auto-detect and rename columns
    column_mapping = {
        'price': 'Market_Price(INR)',
        'selling_price': 'Market_Price(INR)',
        'market_price': 'Market_Price(INR)',
        'brand': 'Brand',
        'make': 'Brand',
        'model': 'Model',
        'year': 'Year',
        'mileage': 'Mileage',
        'km_driven': 'Mileage',
        'fuel': 'Fuel_Type',
        'fuel_type': 'Fuel_Type',
        'transmission': 'Transmission',
        'owner': 'Owner_Type',
        'city': 'City'
    }
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in column_mapping:
            df.rename(columns={col: column_mapping[col_lower]}, inplace=True)
    
    # Data cleaning
    df = df.dropna(subset=['Market_Price(INR)'] if 'Market_Price(INR)' in df.columns else [df.columns[0]])
    
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year'])
        df['Year'] = df['Year'].astype(int)
        df = df[df['Year'] >= 1990]
        df = df[df['Year'] <= datetime.now().year]
    
    if 'Market_Price(INR)' in df.columns:
        df = df[df['Market_Price(INR)'] > 0]
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

try:
    with st.spinner('🔄 Loading and processing data...'):
        df = load_and_clean_data(uploaded_file)
        st.success(f"✅ Successfully loaded {len(df):,} car records!")
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# Validate required columns
if 'Market_Price(INR)' not in df.columns:
    st.error("❌ Price column not found! Please ensure your dataset has a price column.")
    st.stop()

if 'Brand' not in df.columns or 'Model' not in df.columns:
    st.warning("⚠️ Brand/Model columns missing. Limited functionality available.")

# Train multiple models with cross-validation
@st.cache_resource
def train_models(df):
    current_year = datetime.now().year
    df_model = df.copy()
    
    # Feature Engineering
    if 'Year' in df_model.columns:
        df_model['Car_Age'] = current_year - df_model['Year']
        df_model['Age_Squared'] = df_model['Car_Age'] ** 2
    
    if 'Brand' in df_model.columns:
        brand_avg = df_model.groupby('Brand')['Market_Price(INR)'].mean()
        brand_count = df_model.groupby('Brand').size()
        df_model['Brand_Avg_Price'] = df_model['Brand'].map(brand_avg)
        df_model['Brand_Popularity'] = df_model['Brand'].map(brand_count)
    
    if 'Model' in df_model.columns and 'Brand' in df_model.columns:
        model_avg = df_model.groupby(['Brand', 'Model'])['Market_Price(INR)'].mean()
        df_model['Model_Avg_Price'] = df_model.apply(lambda x: model_avg.get((x['Brand'], x['Model']), 0), axis=1)
    
    if 'Mileage' in df_model.columns:
        df_model['Mileage'] = pd.to_numeric(df_model['Mileage'], errors='coerce')
        df_model['Mileage'].fillna(df_model['Mileage'].median(), inplace=True)
        df_model['High_Mileage'] = (df_model['Mileage'] > df_model['Mileage'].median()).astype(int)
    
    # Encode categorical variables
    cat_cols = df_model.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le
    
    # Prepare features
    X = df_model.drop(columns=['Market_Price(INR)'], errors='ignore')
    y = df_model['Market_Price(INR)']
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    best_model = None
    best_score = -np.inf
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        
        results[name] = {
            'model': model,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'accuracy': r2 * 100
        }
        
        if r2 > best_score:
            best_score = r2
            best_model = name
    
    # Feature importance
    if hasattr(results['Random Forest']['model'], 'feature_importances_'):
        importances = results['Random Forest']['model'].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = None
    
    return {
        'results': results,
        'best_model': best_model,
        'scaler': scaler,
        'encoders': encoders,
        'features': X.columns.tolist(),
        'feature_importance': feature_importance,
        'X_test': X_test,
        'y_test': y_test
    }

# Train models
with st.spinner('🎯 Training AI models with cross-validation...'):
    model_data = train_models(df)
    st.session_state.model_trained = True
    st.session_state.model_data = model_data

best_model_name = model_data['best_model']
best_model_results = model_data['results'][best_model_name]

st.success(f"✅ Best Model: {best_model_name} | Accuracy: {best_model_results['accuracy']:.2f}% | MAPE: {best_model_results['mape']:.2f}%")

# ============================================
# PAGES
# ============================================

if page == "🏠 Dashboard":
    st.markdown("<div class='section-header'>📊 BUSINESS DASHBOARD</div>", unsafe_allow_html=True)
    st.markdown("")
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📦 Total Cars", f"{len(df):,}")
    with col2:
        st.metric("🏢 Brands", f"{df['Brand'].nunique() if 'Brand' in df.columns else 'N/A'}")
    with col3:
        avg_price = df['Market_Price(INR)'].mean()
        st.metric("💰 Avg Price", f"₹{avg_price/100000:.2f}L")
    with col4:
        st.metric("🎯 Model Accuracy", f"{best_model_results['accuracy']:.1f}%")
    with col5:
        st.metric("📊 MAPE", f"{best_model_results['mape']:.2f}%")
    
    st.markdown("---")
    
    # Main Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Price Distribution Analysis")
        fig = px.histogram(df, x='Market_Price(INR)', nbins=50, 
                          title='Market Price Distribution',
                          labels={'Market_Price(INR)': 'Price (₹)', 'count': 'Frequency'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Statistical Summary")
        stats = df['Market_Price(INR)'].describe()
        stats_df = pd.DataFrame({
            'Metric': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
            'Value': [
                f"{stats['count']:.0f}",
                f"₹{stats['mean']:,.0f}",
                f"₹{stats['std']:,.0f}",
                f"₹{stats['min']:,.0f}",
                f"₹{stats['25%']:,.0f}",
                f"₹{stats['50%']:,.0f}",
                f"₹{stats['75%']:,.0f}",
                f"₹{stats['max']:,.0f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Brand Analysis
    if 'Brand' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Top 10 Brands by Volume")
            top_brands = df['Brand'].value_counts().head(10)
            fig = px.bar(x=top_brands.values, y=top_brands.index, orientation='h',
                        labels={'x': 'Number of Cars', 'y': 'Brand'},
                        color=top_brands.values, color_continuous_scale='Blues')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💎 Top 10 Premium Brands by Price")
            brand_price = df.groupby('Brand')['Market_Price(INR)'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=brand_price.values, y=brand_price.index, orientation='h',
                        labels={'x': 'Average Price (₹)', 'y': 'Brand'},
                        color=brand_price.values, color_continuous_scale='Greens')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Year-wise Analysis
    if 'Year' in df.columns:
        st.markdown("---")
        st.markdown("### 📅 Year-wise Market Trends")
        
        year_data = df.groupby('Year').agg({
            'Market_Price(INR)': ['mean', 'count']
        }).reset_index()
        year_data.columns = ['Year', 'Avg_Price', 'Count']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=year_data['Year'], y=year_data['Avg_Price'],
                                mode='lines+markers', name='Avg Price',
                                line=dict(color='#667eea', width=3)))
        fig.update_layout(title='Average Price by Year', 
                         xaxis_title='Year', yaxis_title='Average Price (₹)',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Dataset Preview
    st.markdown("---")
    st.markdown("### 📋 Dataset Preview (First 100 Records)")
    st.dataframe(df.head(100), use_container_width=True)

elif page == "💰 Price Prediction":
    st.markdown("<div class='section-header'>💰 AI-POWERED PRICE PREDICTION</div>", unsafe_allow_html=True)
    st.markdown("")
    
    if 'Brand' not in df.columns or 'Model' not in df.columns:
        st.error("❌ Brand/Model columns required for prediction!")
        st.stop()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🚘 Select Car")
        brand = st.selectbox("Brand", sorted(df['Brand'].unique()))
        
        brand_data = df[df['Brand'] == brand]
        models_list = sorted(brand_data['Model'].unique())
        model_name = st.selectbox("Model", models_list)
        
        selected_car_data = brand_data[brand_data['Model'] == model_name]
        
        if len(selected_car_data) == 0:
            st.warning("⚠️ No data for this combination")
            st.stop()
    
    with col2:
        st.markdown("### 📊 Market Reference")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("📦 Cars Available", len(selected_car_data))
        with col_b:
            avg_price = selected_car_data['Market_Price(INR)'].mean()
            st.metric("💰 Market Average", f"₹{avg_price:,.0f}")
        with col_c:
            price_range = selected_car_data['Market_Price(INR)'].max() - selected_car_data['Market_Price(INR)'].min()
            st.metric("📊 Price Range", f"₹{price_range:,.0f}")
    
    st.markdown("---")
    
    # Input form
    sample_car = selected_car_data.iloc[0]
    st.markdown("### 📝 Enter Car Details")
    
    col1, col2, col3, col4 = st.columns(4)
    inputs = {'Brand': brand, 'Model': model_name}
    
    features = model_data['features']
    encoders = model_data['encoders']
    
    with col1:
        if 'Year' in features:
            years = sorted(selected_car_data['Year'].unique(), reverse=True) if 'Year' in selected_car_data.columns else list(range(2024, 1999, -1))
            inputs['Year'] = st.selectbox("Year", years)
    
    with col2:
        if 'Mileage' in features and 'Mileage' in df.columns:
            min_mil = int(selected_car_data['Mileage'].min()) if 'Mileage' in selected_car_data.columns else 0
            max_mil = int(selected_car_data['Mileage'].max()) if 'Mileage' in selected_car_data.columns else 200000
            inputs['Mileage'] = st.number_input("Mileage (km)", min_mil, max_mil, min_mil + 10000)
    
    with col3:
        if 'Fuel_Type' in features and 'Fuel_Type' in df.columns:
            fuel_options = sorted(selected_car_data['Fuel_Type'].unique()) if 'Fuel_Type' in selected_car_data.columns else ['Petrol', 'Diesel']
            inputs['Fuel_Type'] = st.selectbox("Fuel Type", fuel_options)
    
    with col4:
        if 'Transmission' in features and 'Transmission' in df.columns:
            trans_options = sorted(selected_car_data['Transmission'].unique()) if 'Transmission' in selected_car_data.columns else ['Manual', 'Automatic']
            inputs['Transmission'] = st.selectbox("Transmission", trans_options)
    
    # Additional features
    remaining_cols = [f for f in features if f not in inputs and f not in ['Car_Age', 'Brand_Avg_Price', 'Brand_Popularity', 'Model_Avg_Price', 'Age_Squared', 'High_Mileage']]
    
    if remaining_cols:
        st.markdown("### 🔧 Additional Details")
        cols = st.columns(min(4, len(remaining_cols)))
        for idx, col_name in enumerate(remaining_cols):
            with cols[idx % 4]:
                if col_name in encoders and col_name in df.columns:
                    options = sorted(df[col_name].unique())
                    inputs[col_name] = st.selectbox(col_name, options, key=f"add_{col_name}")
                elif col_name in df.columns:
                    inputs[col_name] = st.number_input(col_name, value=float(sample_car.get(col_name, 0)), key=f"add_{col_name}")
    
    st.markdown("---")
    
    if st.button("🔍 Predict Price with AI", type="primary", use_container_width=True):
        # Prepare input
        input_data = inputs.copy()
        current_year = datetime.now().year
        
        # Feature engineering
        if 'Year' in input_data:
            input_data['Car_Age'] = current_year - input_data['Year']
            input_data['Age_Squared'] = input_data['Car_Age'] ** 2
        
        if 'Brand' in input_data:
            brand_avg = df.groupby('Brand')['Market_Price(INR)'].mean()
            brand_count = df.groupby('Brand').size()
            input_data['Brand_Avg_Price'] = brand_avg.get(input_data['Brand'], avg_price)
            input_data['Brand_Popularity'] = brand_count.get(input_data['Brand'], 0)
        
        if 'Model' in input_data and 'Brand' in input_data:
            model_avg = df.groupby(['Brand', 'Model'])['Market_Price(INR)'].mean()
            input_data['Model_Avg_Price'] = model_avg.get((input_data['Brand'], input_data['Model']), avg_price)
        
        if 'Mileage' in input_data:
            median_mileage = df['Mileage'].median() if 'Mileage' in df.columns else 50000
            input_data['High_Mileage'] = int(input_data['Mileage'] > median_mileage)
        
        # Create dataframe
        input_df = pd.DataFrame([input_data])
        
        # Encode categoricals
        for col in encoders:
            if col in input_df.columns:
                try:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str))
                except:
                    input_df[col] = 0
        
        # Add missing features
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder
        input_df = input_df[features]
        
        # Scale and predict
        scaler = model_data['scaler']
        input_scaled = scaler.transform(input_df)
        
        # Get predictions from all models
        predictions = {}
        for name, result in model_data['results'].items():
            pred = result['model'].predict(input_scaled)[0]
            predictions[name] = pred
        
        # Use best model prediction
        best_prediction = predictions[best_model_name]
        
        # Ensemble prediction (weighted average)
        ensemble_pred = (
            0.5 * predictions['Random Forest'] +
            0.3 * predictions['Gradient Boosting'] +
            0.2 * predictions['Linear Regression']
        )
        
        # Market adjustment
        final_price = 0.6 * ensemble_pred + 0.4 * avg_price
        
        # Display results
        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
        st.markdown("## 💰 AI Price Prediction Results")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        min_price = final_price * 0.92
        max_price = final_price * 1.08
        
        with col1:
            st.metric("🔻 Min Price", f"₹{min_price:,.0f}", delta="-8%", delta_color="normal")
        with col2:
            st.metric("💚 Lower Fair", f"₹{final_price*0.96:,.0f}", delta="-4%", delta_color="normal")
        with col3:
            st.metric("⭐ FAIR PRICE", f"₹{final_price:,.0f}", delta="✓ Best", delta_color="off")
        with col4:
            st.metric("💙 Upper Fair", f"₹{final_price*1.04:,.0f}", delta="+4%", delta_color="normal")
        with col5:
            st.metric("🔺 Max Price", f"₹{max_price:,.0f}", delta="+8%", delta_color="normal")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Prediction Breakdown")
            
            breakdown = pd.DataFrame({
                'Model': list(predictions.keys()) + ['Ensemble', 'Market Adj.', '**FINAL**'],
                'Price': [f"₹{p:,.0f}" for p in predictions.values()] + 
                        [f"₹{ensemble_pred:,.0f}", f"₹{avg_price:,.0f}", f"**₹{final_price:,.0f}**"]
            })
            st.dataframe(breakdown, use_container_width=True, hide_index=True)
            
            st.markdown("### 🎯 Model Confidence")
            st.metric("R² Score", f"{best_model_results['r2']:.4f}")
            st.metric("MAPE", f"{best_model_results['mape']:.2f}%")
            st.metric("MAE", f"₹{best_model_results['mae']:,.0f}")
            
            if 'Year' in inputs:
                age = current_year - inputs['Year']
                depreciation = (1 - (age * 0.08)) * 100
                st.metric("Estimated Value Retention", f"{max(20, depreciation):.0f}%")
        
        with col2:
            st.markdown("### 📈 Price Range Visualization")
            
            fig = go.Figure()
            
            categories = ['Min\nPrice', 'Lower\nFair', 'FAIR\nPRICE', 'Upper\nFair', 'Max\nPrice']
            values = [min_price, final_price*0.96, final_price, final_price*1.04, max_price]
            colors = ['#ff6b6b', '#ffd93d', '#4ecdc4', '#95e1d3', '#a8e6cf']
            
            fig.add_trace(go.Bar(
                x=categories,
                y=values,
                marker=dict(color=colors),
                text=[f"₹{v:,.0f}" for v in values],
                textposition='outside'
            ))
            
            fig.add_hline(y=avg_price, line_dash="dash", line_color="red",
                         annotation_text=f"Market Avg: ₹{avg_price:,.0f}")
            
            fig.update_layout(
                title="Price Range Analysis",
                yaxis_title="Price (₹)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 💡 Pricing Insights")
            
            if final_price < avg_price * 0.9:
                st.info("💰 **Great Deal!** This price is below market average.")
            elif final_price > avg_price * 1.1:
                st.warning("⚠️ **Premium Pricing** - Above market average.")
            else:
                st.success("✅ **Fair Market Price** - Aligned with market.")
            
            if 'Mileage' in inputs:
                avg_mileage = selected_car_data['Mileage'].mean() if 'Mileage' in selected_car_data.columns else 50000
                if inputs['Mileage'] < avg_mileage * 0.7:
                    st.success("👍 Low mileage - adds value!")
                elif inputs['Mileage'] > avg
