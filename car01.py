# ======================================================
# SMART CAR PRICING SYSTEM - OPTIMIZED & FAST (UPDATED)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime

# Page config
st.set_page_config(page_title="Smart Car Pricing", layout="wide")

# Title
st.title("🚗 Smart Car Pricing System")
st.markdown("### Fast & Accurate AI Price Prediction")

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
        "🔧 About & Improvements"
    ])
    if st.button("🔄 Retrain Model"):
        st.cache_resource.clear()
        st.session_state.model_trained = False
        st.rerun()

# File Upload
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.info("👆 Upload CSV file to start!")
    st.markdown("### 📋 Sample Format:")
    st.code("""Brand,Model,Year,Mileage,Fuel_Type,Transmission,City,Price
Maruti,Swift,2020,15000,Petrol,Manual,Delhi,550000
Honda,City,2019,20000,Petrol,Automatic,Mumbai,900000""")
    st.stop()

# Load data with caching for speed
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
    for old, new in [('brand', 'Brand'), ('model', 'Model'), ('year', 'Year'), ('mileage', 'Mileage'), ('fuel', 'Fuel_Type'), ('transmission', 'Transmission'), ('city', 'City')]:
        for col in df.columns:
            if old in col.lower() and col != new:
                df = df.rename(columns={col: new})
                break
    
    # Clean data
    df = df.dropna()
    
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year'])
        df['Year'] = df['Year'].astype(int)
    
    return df

try:
    df = load_data(uploaded_file)
    df_clean = df.copy()
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

# Train model only once with caching
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
    
    # Train with tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5]
    }
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid = GridSearchCV(base_model, param_grid, cv=5, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'features': X.columns.tolist(),
        'r2': r2,
        'mae': mae,
        'accuracy': r2 * 100,
        'cv_mean': cv_scores.mean() * 100,
        'importances': importances,
        'best_params': grid.best_params_
    }

# Train model
with st.spinner('🎯 Training model...'):
    model_data = train_model(df_clean)
    st.session_state.model_trained = True
    st.session_state.model = model_data

st.success(f"✅ Model ready! Accuracy: {model_data['accuracy']:.1f}% | CV Accuracy: {model_data['cv_mean']:.1f}% | Best Params: {model_data['best_params']}")

# Get model components
model = model_data['model']
scaler = model_data['scaler']
encoders = model_data['encoders']
features = model_data['features']
importances = model_data['importances']

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
        st.metric("Accuracy", f"{model_data['accuracy']:.1f}%")
    
    st.markdown("---")
    
    # Quick stats
    if 'Brand' in df_clean.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Top 10 Brands")
            top_brands = df_clean['Brand'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(top_brands.index, top_brands.values, color='skyblue')
            ax.set_xlabel('Count')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("### 💰 Price by Brand (Top 10)")
            brand_price = df_clean.groupby('Brand')['Market_Price(INR)'].mean().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(brand_price.index, brand_price.values, color='lightgreen')
            ax.set_xlabel('Avg Price (₹)')
            plt.ticklabel_format(style='plain', axis='x')
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    
    st.subheader("🔑 Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    importances.plot.bar(ax=ax, color='purple')
    ax.set_ylabel('Importance Score')
    ax.set_title('Features Affecting Price')
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    st.subheader("📋 Dataset Sample")
    st.dataframe(df_clean.head(20), use_container_width=True)
    
    with st.expander("📈 Model Details"):
        st.write(f"Algorithm: RandomForestRegressor - Chosen for its ability to handle non-linear relationships, mixed data types, and provide feature importances. It's robust to overfitting with ensemble methods.")
        st.write(f"Evaluation: R2 Score: {model_data['r2']:.3f}, MAE: ₹{model_data['mae']:,.0f}")
        st.write("Hyperparameter Tuning: Used GridSearchCV with 5-fold CV.")
        st.write("Generalization: Tested on unseen data and cross-validated.")

elif page == "💰 Price Prediction":
    st.subheader("💰 Predict Car Price")
    
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
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cars in Dataset", len(selected_car_data))
    with col2:
        avg_price = selected_car_data['Market_Price(INR)'].mean()
        st.metric("Avg Market Price", f"₹{avg_price:,.0f}")
    with col3:
        price_range = selected_car_data['Market_Price(INR)'].max() - selected_car_data['Market_Price(INR)'].min()
        st.metric("Price Range", f"₹{price_range:,.0f}")
    
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
                    # Get unique values for this specific car model
                    unique_vals = selected_car_data[col].unique()
                    options = sorted(unique_vals)
                    
                    # If no specific values, use all brand values
                    if len(options) == 0:
                        options = sorted(brand_data[col].unique())
                    
                    # Set default
                    try:
                        default_idx = options.index(sample_car[col])
                    except ValueError:
                        default_idx = 0
                    inputs[col] = st.selectbox(f"{col}", options, index=default_idx, key=f"inp_{col}")
                else:
                    # Numeric input
                    min_val = float(selected_car_data[col].min())
                    max_val = float(selected_car_data[col].max())
                    default = float(sample_car[col])
                    inputs[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=default, key=f"inp_{col}")
                
                col_idx += 1
    
    # Predict button
    if st.button("🔍 Predict Price", type="primary"):
        # Check for missing inputs
        if any(v is None or (isinstance(v, (int, float)) and np.isnan(v)) for v in inputs.values()):
            st.error("❌ Please fill all fields with valid data!")
            st.stop()
        
        # Prepare input
        input_data = inputs.copy()
        
        # Add engineered features
        current_year = datetime.now().year
        if 'Year' in input_data:
            if input_data['Year'] > current_year or input_data['Year'] < 1900:
                st.error("❌ Invalid Year!")
                st.stop()
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
                except ValueError:
                    st.error(f"❌ Invalid value for {col}!")
                    st.stop()
        
        # Add missing features
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns
        input_df = input_df[features]
        
        # Scale and predict
        try:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.stop()
        
        # Adjust with market average
        final_price = 0.7 * prediction + 0.3 * avg_price
        
        # Display results
        st.markdown("---")
        st.subheader("💰 Price Estimation")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_price = final_price * 0.95
            st.metric("Min Price", f"₹{min_price:,.0f}", delta="-5%")
        
        with col2:
            st.metric("**Fair Price**", f"₹{final_price:,.0f}", delta="✓ Best")
        
        with col3:
            max_price = final_price * 1.05
            st.metric("Max Price", f"₹{max_price:,.0f}", delta="+5%")
        
        with col4:
            st.metric("Confidence", f"{model_data['accuracy']:.0f}%")
        
        # Chart
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Price Breakdown:**")
            st.write(f"• Base Prediction: ₹{prediction:,.0f}")
            st.write(f"• Market Average: ₹{avg_price:,.0f}")
            st.write(f"• Final (Adjusted): ₹{final_price:,.0f}")
            
            if 'Year' in inputs:
                age = current_year - inputs['Year']
                st.write(f"• Car Age: {age} years")
        
        with col2:
            fig, ax = plt.subplots(figsize=(7, 5))
            labels = ['Min', 'Fair', 'Max']
            values = [min_price, final_price, max_price]
            colors = ['#ff6b6b', '#4ecdc4', '#ffe66d']
            ax.bar(labels, values, color=colors, alpha=0.8)
            ax.set_ylabel('Price (₹)')
            ax.set_title('Price Range')
            plt.ticklabel_format(style='plain', axis='y')
            st.pyplot(fig)
            plt.close()
        
        st.balloons()
        
        # Save prediction
        st.session_state.predictions.append({
            'Brand': brand,
            'Model': model_name,
            'Price': f"₹{final_price:,.0f}",
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M")
        })

elif page == "📊 Compare Cars":
    st.subheader("📊 Compare Cars")
    
    num_cars = st.slider("Number of cars", 2, 3, 2)
    
    comparison_data = []
    cols = st.columns(num_cars)
    
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"### Car {i+1}")
            brand = st.selectbox("Brand", sorted(df_clean['Brand'].unique()), key=f"cb{i}")
            models = df_clean[df_clean['Brand'] == brand]['Model'].unique()
            model = st.selectbox("Model", sorted(models), key=f"cm{i}")
            
            car = df_clean[(df_clean['Brand'] == brand) & (df_clean['Model'] == model)].iloc[0]
            comparison_data.append({
                'Brand': brand,
                'Model': model,
                'Price': car['Market_Price(INR)'],
                'Year': car.get('Year', 'N/A')
            })
    
    if st.button("Compare", type="primary"):
        st.markdown("---")
        
        comp_df = pd.DataFrame(comparison_data).T
        comp_df.columns = [f"Car {i+1}" for i in range(num_cars)]
        st.dataframe(comp_df, use_container_width=True)
        
        # Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        cars = [f"{d['Brand']}\n{d['Model']}" for d in comparison_data]
        prices = [d['Price'] for d in comparison_data]
        colors = ['#667eea', '#764ba2', '#f093fb'][:num_cars]
        ax.bar(cars, prices, color=colors, alpha=0.8)
        ax.set_ylabel('Price (₹)')
        ax.set_title('Price Comparison')
        plt.ticklabel_format(style='plain', axis='y')
        st.pyplot(fig)
        plt.close()
        
        best_idx = prices.index(min(prices))
        st.success(f"💰 Best Value: {comparison_data[best_idx]['Brand']} {comparison_data[best_idx]['Model']} - ₹{comparison_data[best_idx]['Price']:,.0f}")

elif page == "🧮 EMI Calculator":
    st.subheader("🧮 EMI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Loan Details")
        price = st.number_input("Car Price (₹)", 100000, 10000000, 1000000, 50000)
        down = st.slider("Down Payment (%)", 0, 50, 20)
        rate = st.slider("Interest Rate (%)", 5.0, 15.0, 9.5, 0.5)
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
        st.markdown("### EMI Summary")
        st.metric("Monthly EMI", f"₹{emi:,.0f}")
        st.metric("Total Payment", f"₹{total:,.0f}")
        st.metric("Total Interest", f"₹{interest:,.0f}")
        st.metric("Loan Amount", f"₹{loan:,.0f}")
        
        # Pie chart
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie([loan, interest], labels=['Principal', 'Interest'], 
               autopct='%1.1f%%', colors=['#4ecdc4', '#ff6b6b'], startangle=90)
        ax.set_title('Payment Breakdown')
        st.pyplot(fig)
        plt.close()

elif page == "🔧 About & Improvements":
    st.subheader("🔧 About the System")
    st.write("This system uses RandomForest for predictions due to its robustness and feature handling. It can be integrated with CRM via API wrappers or CSV exports.")
    
    st.subheader("📈 Future Improvements")
    st.markdown("""
    - Automated market data fetching via APIs.
    - More ML algorithms (e.g., XGBoost).
    - User authentication for dealerships.
    - Mobile app version.
    - Real-time updates and monitoring.
    """)
    st.write("To update model: Upload new CSV and retrain via sidebar button.")

# Footer
st.markdown("---")
if len(st.session_state.predictions) > 0:
    with st.expander("📜 Prediction History"):
        st.dataframe(pd.DataFrame(st.session_state.predictions), use_container_width=True, hide_index=True)

st.markdown("Made with ❤️ | Fast AI Car Pricing")
