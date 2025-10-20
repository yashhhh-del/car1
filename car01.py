# ======================================================
# SMART CAR PRICING SYSTEM - ENHANCED VERSION
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import io

# Page config
st.set_page_config(page_title="Smart Car Pricing", layout="wide")

# Title
st.title("🚗 Smart Car Pricing System - Enhanced")
st.markdown("### Fast & Accurate AI Price Prediction with Advanced Analytics")

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None

# Sidebar
with st.sidebar:
    st.title("📊 Navigation")
    page = st.radio("Select Page", [
        "🏠 Home",
        "💰 Price Prediction",
        "📊 Compare Cars",
        "🧮 EMI Calculator",
        "📈 Analytics Dashboard",
        "📉 Depreciation Analysis"
    ])
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Stats")

# File Upload
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.info("👆 Upload CSV file to start!")
    st.markdown("### 📋 Sample Format:")
    st.code("""Brand,Model,Year,Mileage,Fuel_Type,Transmission,Price
Maruti,Swift,2020,15000,Petrol,Manual,550000
Honda,City,2019,20000,Petrol,Automatic,900000""")
    st.stop()

# Load data with validation
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    
    # Auto-detect price column
    price_col = None
    for col in df.columns:
        if 'price' in col.lower():
            price_col = col
            break
    
    if price_col and price_col != 'Market_Price(INR)':
        df = df.rename(columns={price_col: 'Market_Price(INR)'})
    
    # Auto-detect other columns
    for old, new in [('brand', 'Brand'), ('model', 'Model'), ('year', 'Year')]:
        for col in df.columns:
            if old in col.lower() and col != new:
                df = df.rename(columns={col: new})
                break
    
    # Data Quality Checks
    initial_rows = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    
    # Clean data
    df = df.dropna()
    missing_removed = initial_rows - duplicates_removed - len(df)
    
    # Validate Year
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year'])
        df['Year'] = df['Year'].astype(int)
        # Keep only valid years
        df = df[(df['Year'] >= 1980) & (df['Year'] <= datetime.now().year)]
    
    # Outlier detection for price
    if 'Market_Price(INR)' in df.columns:
        Q1 = df['Market_Price(INR)'].quantile(0.01)
        Q3 = df['Market_Price(INR)'].quantile(0.99)
        IQR = Q3 - Q1
        df = df[(df['Market_Price(INR)'] >= Q1) & (df['Market_Price(INR)'] <= Q3)]
    
    outliers_removed = initial_rows - duplicates_removed - missing_removed - len(df)
    
    quality_report = {
        'initial': initial_rows,
        'duplicates': duplicates_removed,
        'missing': missing_removed,
        'outliers': outliers_removed,
        'final': len(df)
    }
    
    return df, quality_report

try:
    df_clean, quality_report = load_data(uploaded_file)
    
    # Show data quality report
    with st.expander("📊 Data Quality Report"):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Initial Rows", quality_report['initial'])
        col2.metric("Duplicates Removed", quality_report['duplicates'], delta_color="inverse")
        col3.metric("Missing Values", quality_report['missing'], delta_color="inverse")
        col4.metric("Outliers Removed", quality_report['outliers'], delta_color="inverse")
        col5.metric("Clean Data", quality_report['final'], delta="✓")
    
    st.success(f"✅ Loaded {len(df_clean)} cars successfully!")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# Check required columns
if 'Market_Price(INR)' not in df_clean.columns:
    st.error("❌ Price column not found!")
    st.stop()

if 'Brand' not in df_clean.columns or 'Model' not in df_clean.columns:
    st.warning("⚠️ Brand/Model columns missing. Some features won't work.")

# Train model with enhanced metrics
@st.cache_resource
def train_model(df):
    # Feature engineering
    current_year = datetime.now().year
    df_model = df.copy()
    
    if 'Year' in df_model.columns:
        df_model['Car_Age'] = current_year - df_model['Year']
    
    if 'Brand' in df_model.columns:
        brand_avg = df_model.groupby('Brand')['Market_Price(INR)'].mean()
        df_model['Brand_Avg_Price'] = df_model['Brand'].map(brand_avg)
    
    # Encode categorical
    cat_cols = df_model.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le
    
    # Prepare features
    X = df_model.drop(columns=['Market_Price(INR)'], errors='ignore')
    y = df_model['Market_Price(INR)']
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train with cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate with multiple metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'features': X.columns.tolist(),
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'cv_scores': cv_scores,
        'accuracy': r2 * 100,
        'feature_importance': feature_importance,
        'y_test': y_test,
        'y_pred': y_pred
    }

# Train model
with st.spinner('🎯 Training advanced model...'):
    model_data = train_model(df_clean)
    st.session_state.model_trained = True
    st.session_state.model = model_data

# Enhanced model metrics display
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy (R²)", f"{model_data['accuracy']:.1f}%")
col2.metric("RMSE", f"₹{model_data['rmse']:,.0f}")
col3.metric("MAPE", f"{model_data['mape']:.1f}%")
col4.metric("CV Score", f"{model_data['cv_scores'].mean():.2f}")

# Get model components
model = model_data['model']
scaler = model_data['scaler']
encoders = model_data['encoders']
features = model_data['features']

# ============================================
# PAGES
# ============================================

if page == "🏠 Home":
    st.subheader("📊 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cars", f"{len(df_clean):,}")
    with col2:
        st.metric("Brands", f"{df_clean['Brand'].nunique()}")
    with col3:
        st.metric("Avg Price", f"₹{df_clean['Market_Price(INR)'].mean()/100000:.1f}L")
    with col4:
        st.metric("Model Accuracy", f"{model_data['accuracy']:.1f}%")
    
    st.markdown("---")
    
    # Interactive plots with Plotly
    if 'Brand' in df_clean.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Top 10 Brands by Count")
            top_brands = df_clean['Brand'].value_counts().head(10)
            fig = px.bar(x=top_brands.values, y=top_brands.index, orientation='h',
                        labels={'x': 'Count', 'y': 'Brand'},
                        color=top_brands.values, color_continuous_scale='Blues')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Average Price by Brand (Top 10)")
            brand_price = df_clean.groupby('Brand')['Market_Price(INR)'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=brand_price.values, y=brand_price.index, orientation='h',
                        labels={'x': 'Avg Price (₹)', 'y': 'Brand'},
                        color=brand_price.values, color_continuous_scale='Greens')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Price Distribution
    st.markdown("---")
    st.markdown("### 📊 Price Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df_clean, x='Market_Price(INR)', nbins=50,
                          labels={'Market_Price(INR)': 'Price (₹)'},
                          title='Price Distribution')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df_clean, y='Market_Price(INR)',
                    labels={'Market_Price(INR)': 'Price (₹)'},
                    title='Price Box Plot')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Dataset Sample")
    st.dataframe(df_clean.head(20), use_container_width=True)

elif page == "💰 Price Prediction":
    st.subheader("💰 Enhanced Price Prediction")
    
    if 'Brand' not in df_clean.columns or 'Model' not in df_clean.columns:
        st.error("❌ Brand/Model columns required!")
        st.stop()
    
    # Brand selection
    brand = st.selectbox("🚘 Select Brand", sorted(df_clean['Brand'].unique()))
    
    # Filter models for selected brand
    brand_data = df_clean[df_clean['Brand'] == brand]
    models_list = sorted(brand_data['Model'].unique())
    
    # Model selection
    model_name = st.selectbox("🔧 Select Model", models_list)
    
    # Filter data for selected brand and model
    selected_car_data = brand_data[brand_data['Model'] == model_name]
    
    if len(selected_car_data) == 0:
        st.warning("No data found for this combination")
        st.stop()
    
    # Show market reference
    st.markdown("---")
    st.subheader(f"📊 Market Data: {brand} {model_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cars in Dataset", len(selected_car_data))
    with col2:
        avg_price = selected_car_data['Market_Price(INR)'].mean()
        st.metric("Avg Market Price", f"₹{avg_price:,.0f}")
    with col3:
        price_range = selected_car_data['Market_Price(INR)'].max() - selected_car_data['Market_Price(INR)'].min()
        st.metric("Price Range", f"₹{price_range:,.0f}")
    with col4:
        median_price = selected_car_data['Market_Price(INR)'].median()
        st.metric("Median Price", f"₹{median_price:,.0f}")
    
    # Get sample car
    sample_car = selected_car_data.iloc[0]
    
    st.markdown("---")
    st.markdown("### 📝 Car Details")
    
    # Dynamic inputs based on selected car
    col1, col2, col3 = st.columns(3)
    inputs = {}
    col_idx = 0
    
    # Get unique values for this brand-model combination
    for col in features:
        if col in ['Car_Age', 'Brand_Avg_Price']:
            continue
        
        if col in sample_car.index:
            with [col1, col2, col3][col_idx % 3]:
                # Get options specific to this brand-model
                if col in encoders:
                    unique_vals = selected_car_data[col].unique()
                    options = sorted(unique_vals)
                    
                    if len(options) == 0:
                        options = sorted(brand_data[col].unique())
                    
                    default = sample_car[col] if sample_car[col] in options else options[0]
                    inputs[col] = st.selectbox(f"{col}", options, index=options.index(default), key=f"inp_{col}")
                else:
                    min_val = float(selected_car_data[col].min())
                    max_val = float(selected_car_data[col].max())
                    default = float(sample_car[col])
                    inputs[col] = st.number_input(f"{col}", min_val, max_val, default, key=f"inp_{col}")
                
                col_idx += 1
    
    # Add condition factor
    st.markdown("---")
    condition = st.select_slider(
        "🔧 Car Condition",
        options=["Poor", "Fair", "Good", "Excellent"],
        value="Good"
    )
    
    condition_multiplier = {
        "Poor": 0.75,
        "Fair": 0.90,
        "Good": 1.0,
        "Excellent": 1.10
    }
    
    # Predict button
    if st.button("🔍 Predict Price", type="primary"):
        # Prepare input
        input_data = inputs.copy()
        
        # Add engineered features
        current_year = datetime.now().year
        if 'Year' in input_data:
            input_data['Car_Age'] = current_year - input_data['Year']
        
        if 'Brand' in input_data:
            brand_avg = df_clean.groupby('Brand')['Market_Price(INR)'].mean()
            input_data['Brand_Avg_Price'] = brand_avg.get(input_data['Brand'], avg_price)
        
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
        
        # Reorder columns
        input_df = input_df[features]
        
        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        
        # Adjust with market average and condition
        base_price = 0.7 * prediction + 0.3 * avg_price
        final_price = base_price * condition_multiplier[condition]
        
        # Calculate confidence interval
        confidence = model_data['accuracy'] / 100
        lower_bound = final_price * (1 - (1 - confidence) * 0.15)
        upper_bound = final_price * (1 + (1 - confidence) * 0.15)
        
        # Display results
        st.markdown("---")
        st.subheader("💰 Price Estimation with Confidence")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Lower Bound", f"₹{lower_bound:,.0f}", delta="Conservative")
        
        with col2:
            st.metric("**Fair Price**", f"₹{final_price:,.0f}", delta="✓ Best Estimate")
        
        with col3:
            st.metric("Upper Bound", f"₹{upper_bound:,.0f}", delta="Optimistic")
        
        with col4:
            st.metric("Confidence", f"{model_data['accuracy']:.0f}%", delta=f"±{(upper_bound-lower_bound)/2:,.0f}")
        
        # Charts
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**💡 Price Breakdown:**")
            st.write(f"• Base ML Prediction: ₹{prediction:,.0f}")
            st.write(f"• Market Average: ₹{avg_price:,.0f}")
            st.write(f"• Adjusted Price: ₹{base_price:,.0f}")
            st.write(f"• Condition Factor: {condition} ({condition_multiplier[condition]}x)")
            st.write(f"• **Final Price: ₹{final_price:,.0f}**")
            
            if 'Year' in inputs:
                age = current_year - inputs['Year']
                st.write(f"• Car Age: {age} years")
                depreciation = ((avg_price - final_price) / avg_price * 100) if avg_price > 0 else 0
                st.write(f"• Depreciation: {depreciation:.1f}%")
        
        with col2:
            # Interactive price range chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Lower', 'Fair Price', 'Upper'],
                y=[lower_bound, final_price, upper_bound],
                marker_color=['#ff6b6b', '#4ecdc4', '#ffe66d'],
                text=[f"₹{lower_bound:,.0f}", f"₹{final_price:,.0f}", f"₹{upper_bound:,.0f}"],
                textposition='auto'
            ))
            fig.update_layout(
                title='Price Range with Confidence',
                yaxis_title='Price (₹)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.balloons()
        
        # Save prediction
        st.session_state.predictions.append({
            'Brand': brand,
            'Model': model_name,
            'Condition': condition,
            'Fair Price': f"₹{final_price:,.0f}",
            'Lower': f"₹{lower_bound:,.0f}",
            'Upper': f"₹{upper_bound:,.0f}",
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M")
        })

elif page == "📊 Compare Cars":
    st.subheader("📊 Advanced Car Comparison")
    
    num_cars = st.slider("Number of cars to compare", 2, 4, 2)
    
    comparison_data = []
    cols = st.columns(num_cars)
    
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"### Car {i+1}")
            brand = st.selectbox("Brand", sorted(df_clean['Brand'].unique()), key=f"cb{i}")
            models = df_clean[df_clean['Brand'] == brand]['Model'].unique()
            model = st.selectbox("Model", sorted(models), key=f"cm{i}")
            
            car_data = df_clean[(df_clean['Brand'] == brand) & (df_clean['Model'] == model)]
            if len(car_data) > 0:
                car = car_data.iloc[0]
                avg_price = car_data['Market_Price(INR)'].mean()
                comparison_data.append({
                    'Brand': brand,
                    'Model': model,
                    'Price': car['Market_Price(INR)'],
                    'Avg Price': avg_price,
                    'Year': car.get('Year', 'N/A'),
                    'Count': len(car_data)
                })
    
    if st.button("📊 Compare Now", type="primary"):
        st.markdown("---")
        
        # Comparison table
        comp_df = pd.DataFrame(comparison_data).T
        comp_df.columns = [f"Car {i+1}" for i in range(num_cars)]
        st.dataframe(comp_df, use_container_width=True)
        
        # Interactive comparison chart
        fig = go.Figure()
        
        cars = [f"{d['Brand']}\n{d['Model']}" for d in comparison_data]
        prices = [d['Avg Price'] for d in comparison_data]
        colors = ['#667eea', '#764ba2', '#f093fb', '#fccb90'][:num_cars]
        
        fig.add_trace(go.Bar(
            x=cars,
            y=prices,
            marker_color=colors,
            text=[f"₹{p:,.0f}" for p in prices],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Price Comparison',
            yaxis_title='Average Price (₹)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Best value analysis
        best_idx = prices.index(min(prices))
        worst_idx = prices.index(max(prices))
        
        col1, col2 = st.columns(2)
        col1.success(f"💰 Best Value: {comparison_data[best_idx]['Brand']} {comparison_data[best_idx]['Model']} - ₹{comparison_data[best_idx]['Avg Price']:,.0f}")
        col2.info(f"💎 Premium Option: {comparison_data[worst_idx]['Brand']} {comparison_data[worst_idx]['Model']} - ₹{comparison_data[worst_idx]['Avg Price']:,.0f}")
        
        # Export comparison
        if st.button("📥 Download Comparison"):
            comp_export = pd.DataFrame(comparison_data)
            csv = comp_export.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"car_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

elif page == "🧮 EMI Calculator":
    st.subheader("🧮 Advanced EMI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Loan Details")
        price = st.number_input("Car Price (₹)", 100000, 10000000, 1000000, 50000)
        down = st.slider("Down Payment (%)", 0, 50, 20)
        rate = st.slider("Interest Rate (% per annum)", 5.0, 15.0, 9.5, 0.5)
        tenure = st.slider("Tenure (years)", 1, 7, 5)
    
    loan = price * (1 - down/100)
    months = tenure * 12
    r = rate / (12 * 100)
    
    if loan > 0 and r > 0:
        emi = loan * r * ((1 + r)**months) / (((1 + r)**months) - 1)
    else:
        emi = loan / months if months > 0 else 0
    
    total = emi * months
    interest = total - loan
    
    with col2:
        st.markdown("### 💳 EMI Summary")
        st.metric("Monthly EMI", f"₹{emi:,.0f}")
        st.metric("Total Payment", f"₹{total:,.0f}")
        st.metric("Total Interest", f"₹{interest:,.0f}")
        st.metric("Loan Amount", f"₹{loan:,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Principal', 'Interest'],
            values=[loan, interest],
            hole=0.4,
            marker_colors=['#4ecdc4', '#ff6b6b']
        )])
        fig.update_layout(title='Payment Breakdown', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Amortization schedule
        st.markdown("### 📊 Payment Schedule")
        schedule_data = []
        balance = loan
        for month in range(1, min(months + 1, 61)):  # Show first 5 years max
            interest_payment = balance * r
            principal_payment = emi - interest_payment
            balance -= principal_payment
            if month % 12 == 0:  # Show yearly
                schedule_data.append({
                    'Year': month // 12,
                    'EMI': emi,
                    'Principal': principal_payment * 12,
                    'Interest': interest_payment * 12,
                    'Balance': max(0, balance)
                })
        
        schedule_df = pd.DataFrame(schedule_data)
        st.dataframe(schedule_df.style.format({
            'EMI': '₹{:,.0f}',
            'Principal': '₹{:,.0f}',
            'Interest': '₹{:,.0f}',
            'Balance': '₹{:,.0f}'
        }), use_container_width=True)

elif page == "📈 Analytics Dashboard":
    st.subheader("📈 Advanced Analytics Dashboard")
    
    # Feature Importance
    st.markdown("### 🎯 Feature Importance")
    fig = px.bar(
        model_data['feature_importance'].head(10),
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Most Important Features',
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Model Performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Model Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['R² Score', 'MAE', 'RMSE', 'MAPE', 'CV Mean'],
            'Value': [
                f"{model_data['r2']:.4f}",
                f"₹{model_data['mae']:,.0f}",
                f"₹{model_data['rmse']:,.0f}",
                f"{model_data['mape']:.2f}%",
                f"{model_data['cv_scores'].mean():.4f}"
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 🎲 Cross-Validation Scores")
        cv_df = pd.DataFrame({
            'Fold': range(1, 6),
            'R² Score': model_data['cv_scores']
        })
        fig = px.line(cv_df, x='Fold', y='R² Score', markers=True,
                     title='Cross-Validation Performance')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Predicted vs Actual
    st.markdown("### 🎯 Predicted vs Actual Prices")
    pred_actual_df = pd.DataFrame({
        'Actual': model_data['y_test'],
        'Predicted': model_data['y_pred']
    })
    
    fig = px.scatter(pred_actual_df, x='Actual', y='Predicted',
                    title='Model Prediction Accuracy',
                    labels={'Actual': 'Actual Price (₹)', 'Predicted': 'Predicted Price (₹)'},
                    trendline='ols')
    
    # Add perfect prediction line
    max_val = max(pred_actual_df['Actual'].max(), pred_actual_df['Predicted'].max())
    min_val = min(pred_actual_df['Actual'].min(), pred_actual_df['Predicted'].min())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                            mode='lines', name='Perfect Prediction',
                            line=dict(color='red', dash='dash')))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation Heatmap
    if len(df_clean.select_dtypes(include=[np.number]).columns) > 2:
        st.markdown("### 🔥 Feature Correlation Heatmap")
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        corr_matrix = df_clean[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix,
                       labels=dict(color="Correlation"),
                       x=corr_matrix.columns,
                       y=corr_matrix.columns,
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

elif page == "📉 Depreciation Analysis":
    st.subheader("📉 Car Depreciation Analysis")
    
    if 'Year' not in df_clean.columns:
        st.error("❌ Year column required for depreciation analysis!")
        st.stop()
    
    st.markdown("### 🚗 Select Car for Depreciation Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        dep_brand = st.selectbox("Brand", sorted(df_clean['Brand'].unique()), key="dep_brand")
    
    with col2:
        dep_models = df_clean[df_clean['Brand'] == dep_brand]['Model'].unique()
        dep_model = st.selectbox("Model", sorted(dep_models), key="dep_model")
    
    # Get data for selected car
    dep_data = df_clean[(df_clean['Brand'] == dep_brand) & (df_clean['Model'] == dep_model)].copy()
    
    if len(dep_data) == 0:
        st.warning("No data available for depreciation analysis")
        st.stop()
    
    # Calculate depreciation
    current_year = datetime.now().year
    dep_data['Car_Age'] = current_year - dep_data['Year']
    dep_data = dep_data[dep_data['Car_Age'] >= 0].sort_values('Car_Age')
    
    if len(dep_data) < 2:
        st.warning("Insufficient data for depreciation trend analysis")
        st.stop()
    
    # Group by age and calculate average price
    age_price = dep_data.groupby('Car_Age')['Market_Price(INR)'].agg(['mean', 'count']).reset_index()
    age_price = age_price[age_price['count'] >= 1]  # At least 1 car per age
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if len(age_price) > 0:
        newest_price = age_price[age_price['Car_Age'] == age_price['Car_Age'].min()]['mean'].values[0]
        oldest_price = age_price[age_price['Car_Age'] == age_price['Car_Age'].max()]['mean'].values[0]
        total_depreciation = ((newest_price - oldest_price) / newest_price * 100) if newest_price > 0 else 0
        
        col1.metric("New Car Price (Est)", f"₹{newest_price:,.0f}")
        col2.metric("Current Avg Price", f"₹{dep_data['Market_Price(INR)'].mean():,.0f}")
        col3.metric("Total Depreciation", f"{total_depreciation:.1f}%", delta_color="inverse")
        col4.metric("Avg Age", f"{dep_data['Car_Age'].mean():.1f} years")
    
    st.markdown("---")
    
    # Depreciation curve
    st.markdown("### 📉 Depreciation Trend Over Time")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=age_price['Car_Age'],
        y=age_price['mean'],
        mode='lines+markers',
        name='Average Price',
        line=dict(color='#4ecdc4', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title=f'{dep_brand} {dep_model} - Price vs Age',
        xaxis_title='Car Age (years)',
        yaxis_title='Average Price (₹)',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Yearly depreciation rate
    st.markdown("### 📊 Year-over-Year Depreciation Rate")
    
    if len(age_price) > 1:
        age_price['Depreciation_Rate'] = age_price['mean'].pct_change() * -100
        age_price_filtered = age_price[age_price['Car_Age'] > 0].copy()
        
        if len(age_price_filtered) > 0:
            fig = px.bar(age_price_filtered, x='Car_Age', y='Depreciation_Rate',
                        title='Depreciation Rate by Year',
                        labels={'Car_Age': 'Car Age (years)', 'Depreciation_Rate': 'Depreciation Rate (%)'},
                        color='Depreciation_Rate',
                        color_continuous_scale='Reds')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Future value calculator
    st.markdown("### 🔮 Future Value Estimator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_car_year = st.number_input("Your Car's Year", 
                                          int(dep_data['Year'].min()), 
                                          current_year, 
                                          current_year - 2)
        years_to_predict = st.slider("Years into Future", 1, 10, 3)
    
    with col2:
        car_age_now = current_year - current_car_year
        future_age = car_age_now + years_to_predict
        
        # Simple depreciation model (exponential decay)
        if len(age_price) > 1:
            avg_annual_depreciation = age_price['Depreciation_Rate'].mean() / 100
            current_estimated_price = newest_price * ((1 - avg_annual_depreciation) ** car_age_now)
            future_price = current_estimated_price * ((1 - avg_annual_depreciation) ** years_to_predict)
            
            st.metric("Current Estimated Value", f"₹{current_estimated_price:,.0f}")
            st.metric(f"Value after {years_to_predict} years", f"₹{future_price:,.0f}",
                     delta=f"-₹{current_estimated_price - future_price:,.0f}")
            
            # ROI Analysis
            value_loss = current_estimated_price - future_price
            annual_loss = value_loss / years_to_predict
            st.metric("Estimated Annual Loss", f"₹{annual_loss:,.0f}")
    
    st.markdown("---")
    
    # Best time to sell analysis
    st.markdown("### ⏰ Best Time to Sell")
    
    if len(age_price) > 2:
        # Find age with best value retention
        age_price['Value_Retention'] = (age_price['mean'] / newest_price * 100) if newest_price > 0 else 0
        age_price['Value_Per_Year'] = age_price['mean'] / (age_price['Car_Age'] + 1)
        
        best_age_idx = age_price['Value_Per_Year'].idxmax()
        best_age = age_price.loc[best_age_idx, 'Car_Age']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **💡 Recommendation:**
            
            Based on historical data, the optimal time to sell a {dep_brand} {dep_model} 
            is around **{int(best_age)} years** of age, where you get the best value 
            retention per year of ownership.
            
            - Value retention: **{age_price.loc[best_age_idx, 'Value_Retention']:.1f}%**
            - Average price at {int(best_age)} years: **₹{age_price.loc[best_age_idx, 'mean']:,.0f}**
            """)
        
        with col2:
            # Value retention chart
            fig = px.line(age_price, x='Car_Age', y='Value_Retention',
                         title='Value Retention Over Time',
                         labels={'Car_Age': 'Car Age (years)', 'Value_Retention': 'Value Retention (%)'},
                         markers=True)
            fig.add_hline(y=50, line_dash="dash", line_color="red", 
                         annotation_text="50% Mark")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

# Footer with enhanced features
st.markdown("---")

# Prediction History
if len(st.session_state.predictions) > 0:
    with st.expander("📜 Prediction History"):
        pred_df = pd.DataFrame(st.session_state.predictions)
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        
        # Export predictions
        if st.button("📥 Download Predictions"):
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# Export full dataset
with st.expander("📊 Export Dataset"):
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Full Dataset"):
            csv = df_clean.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"car_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📥 Download Model Metrics"):
            metrics_export = pd.DataFrame({
                'Metric': ['R² Score', 'MAE', 'RMSE', 'MAPE', 'CV Mean Score', 'CV Std'],
                'Value': [
                    model_data['r2'],
                    model_data['mae'],
                    model_data['rmse'],
                    model_data['mape'],
                    model_data['cv_scores'].mean(),
                    model_data['cv_scores'].std()
                ]
            })
            csv = metrics_export.to_csv(index=False)
            st.download_button(
                label="Download Metrics",
                data=csv,
                file_name=f"model_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

st.markdown("---")
st.markdown("Made with ❤️ | Enhanced Smart Car Pricing System | Powered by Advanced ML")
