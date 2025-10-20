# ======================================================
# SMART CAR PRICING SYSTEM - ACCURATE PREDICTION
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime

# Page config
st.set_page_config(page_title="Smart Car Pricing", layout="wide")

# Title
st.title("🚗 Smart Car Pricing System")
st.markdown("### AI-Powered Accurate Price Prediction")

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Sidebar
with st.sidebar:
    st.title("📊 Navigation")
    page = st.radio("Select Page", [
        "🏠 Home",
        "💰 Price Prediction",
        "📊 Compare Cars",
        "🧮 EMI Calculator",
        "📈 Market Insights"
    ])

# File Upload
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.info("👆 Please upload a CSV file to get started!")
    st.markdown("---")
    st.markdown("### 📋 Sample CSV Format:")
    st.code("""Brand,Model,Year,Mileage,Fuel_Type,Transmission,Price
Maruti,Swift,2020,15000,Petrol,Manual,550000
Honda,City,2019,20000,Petrol,Automatic,900000
Hyundai,Creta,2021,10000,Diesel,Manual,1200000""")
    st.stop()

# Load data
try:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading file: {e}")
    st.stop()

# Show columns
with st.expander("📋 View Uploaded Columns"):
    st.write("**Columns in your file:**")
    for i, col in enumerate(df.columns, 1):
        st.write(f"{i}. {col}")

# Auto-detect price column
price_col = None
for col in df.columns:
    if 'price' in col.lower():
        price_col = col
        break

if price_col is None:
    st.error("❌ Price column not found! Please ensure CSV has a 'Price' column")
    st.stop()

if price_col != 'Market_Price(INR)':
    df = df.rename(columns={price_col: 'Market_Price(INR)'})
    st.info(f"✅ Using '{price_col}' as price column")

# Auto-detect other columns
for old, new in [('brand', 'Brand'), ('model', 'Model'), ('year', 'Year')]:
    for col in df.columns:
        if old in col.lower() and col != new:
            df = df.rename(columns={col: new})
            break

# Clean data
df_clean = df.dropna()

if 'Year' in df_clean.columns:
    df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Year'])
    df_clean['Year'] = df_clean['Year'].astype(int)

# Feature Engineering for Better Accuracy
st.info("🔧 Engineering features for better predictions...")

# Add car age
current_year = datetime.now().year
if 'Year' in df_clean.columns:
    df_clean['Car_Age'] = current_year - df_clean['Year']

# Add price per year depreciation
if 'Year' in df_clean.columns:
    df_clean['Price_Per_Year'] = df_clean['Market_Price(INR)'] / (current_year - df_clean['Year'] + 1)

# Brand popularity score
if 'Brand' in df_clean.columns:
    brand_counts = df_clean['Brand'].value_counts()
    df_clean['Brand_Popularity'] = df_clean['Brand'].map(brand_counts)

# Average brand price
if 'Brand' in df_clean.columns:
    brand_avg_price = df_clean.groupby('Brand')['Market_Price(INR)'].mean()
    df_clean['Brand_Avg_Price'] = df_clean['Brand'].map(brand_avg_price)

# Encode categorical columns
cat_cols = df_clean.select_dtypes(include=['object']).columns
encoders = {}
df_encoded = df_clean.copy()

for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_clean[col].astype(str))
    encoders[col] = le

# Prepare features
X = df_encoded.drop(columns=['Market_Price(INR)'], errors='ignore')
y = df_encoded['Market_Price(INR)']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
feature_columns = X.columns.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train ensemble of models for better accuracy
with st.spinner('🎯 Training advanced models for accurate predictions...'):
    
    # Multiple models with different strengths
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ),
        'Ridge': Ridge(alpha=1.0)
    }
    
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Cross-validation for reliability
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        
        model_scores[name] = {
            'R2 Score': r2,
            'MAE': mae,
            'CV Score': cv_scores.mean(),
            'Accuracy': r2 * 100
        }
    
    # Select best model
    best_model_name = max(model_scores.items(), key=lambda x: x[1]['R2 Score'])[0]
    best_model = trained_models[best_model_name]
    
    # Create ensemble prediction (averaging top 2 models)
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['R2 Score'], reverse=True)
    top_models = [trained_models[name] for name, _ in sorted_models[:2]]

st.success(f"✅ Model trained! Best: {best_model_name} | Accuracy: {model_scores[best_model_name]['Accuracy']:.1f}%")

# ============================================
# PAGES
# ============================================

if page == "🏠 Home":
    st.subheader("📊 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cars", f"{len(df_clean):,}")
    with col2:
        st.metric("Unique Brands", f"{df_clean['Brand'].nunique()}")
    with col3:
        avg_price = df_clean['Market_Price(INR)'].mean()
        st.metric("Avg. Price", f"₹{avg_price/100000:.1f}L")
    with col4:
        st.metric("Model Accuracy", f"{model_scores[best_model_name]['Accuracy']:.1f}%")
    
    st.markdown("---")
    
    # Model Performance
    st.subheader("🎯 Model Performance")
    perf_df = pd.DataFrame(model_scores).T
    st.dataframe(perf_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    
    st.markdown("---")
    
    # Price distribution
    if 'Brand' in df_clean.columns:
        st.subheader("📈 Price Distribution by Brand")
        top_brands = df_clean['Brand'].value_counts().head(8).index
        filtered = df_clean[df_clean['Brand'].isin(top_brands)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=filtered, x='Brand', y='Market_Price(INR)', ax=ax, palette='Set2')
        ax.set_title('Price Range by Top 8 Brands')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
    
    # Data sample
    st.subheader("📋 Dataset Sample")
    st.dataframe(df_clean.head(10), use_container_width=True)

elif page == "💰 Price Prediction":
    st.subheader("💰 Accurate Price Prediction")
    
    st.info(f"🎯 Using {best_model_name} with {model_scores[best_model_name]['Accuracy']:.1f}% accuracy")
    
    if 'Brand' not in df_clean.columns or 'Model' not in df_clean.columns:
        st.error("❌ Brand and Model columns required!")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox("🚘 Select Brand", sorted(df_clean['Brand'].unique()))
    
    with col2:
        models_list = df_clean[df_clean['Brand'] == brand]['Model'].unique()
        model_name = st.selectbox("🔧 Select Model", sorted(models_list))
    
    # Get similar cars for reference
    similar_cars = df_clean[(df_clean['Brand'] == brand) & (df_clean['Model'] == model_name)]
    
    if len(similar_cars) > 0:
        st.markdown("---")
        st.subheader("📊 Market Reference")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Similar Cars Found", len(similar_cars))
        with col2:
            avg_market_price = similar_cars['Market_Price(INR)'].mean()
            st.metric("Avg Market Price", f"₹{avg_market_price:,.0f}")
        with col3:
            price_range = similar_cars['Market_Price(INR)'].max() - similar_cars['Market_Price(INR)'].min()
            st.metric("Price Range", f"₹{price_range:,.0f}")
    
    # Get sample data
    sample = similar_cars.iloc[0] if len(similar_cars) > 0 else df_clean.iloc[0]
    
    st.markdown("---")
    st.markdown("### 📝 Enter Car Details")
    
    # Input features in columns
    col1, col2, col3 = st.columns(3)
    inputs = {}
    col_idx = 0
    
    for col in feature_columns:
        if col in ['Car_Age', 'Price_Per_Year', 'Brand_Popularity', 'Brand_Avg_Price']:
            continue  # Skip engineered features
        
        if col in sample.index:
            with [col1, col2, col3][col_idx % 3]:
                if col in encoders:
                    options = sorted(df_clean[col].unique())
                    default = sample[col] if sample[col] in options else options[0]
                    inputs[col] = st.selectbox(f"{col}", options, index=options.index(default))
                else:
                    min_val = float(df_clean[col].min())
                    max_val = float(df_clean[col].max())
                    default = float(sample[col])
                    inputs[col] = st.number_input(f"{col}", min_val, max_val, default)
                col_idx += 1
    
    if st.button("🔍 Predict Accurate Price", type="primary"):
        # Prepare input with engineered features
        input_data = inputs.copy()
        
        # Add engineered features
        if 'Year' in input_data:
            input_data['Car_Age'] = current_year - input_data['Year']
            input_data['Price_Per_Year'] = 500000 / (input_data['Car_Age'] + 1)  # Placeholder
        
        if 'Brand' in input_data:
            input_data['Brand_Popularity'] = brand_counts.get(input_data['Brand'], 0)
            input_data['Brand_Avg_Price'] = brand_avg_price.get(input_data['Brand'], 0)
        
        input_df = pd.DataFrame([input_data])
        
        # Encode categoricals
        for col in encoders:
            if col in input_df.columns:
                try:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str))
                except:
                    input_df[col] = 0
        
        # Ensure all features present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[feature_columns]
        
        # Scale and predict
        input_scaled = scaler.transform(input_df)
        
        # Ensemble prediction (average of top models)
        predictions = [model.predict(input_scaled)[0] for model in top_models]
        final_prediction = np.mean(predictions)
        prediction_std = np.std(predictions)
        
        # Get similar car prices for validation
        if len(similar_cars) > 0:
            market_avg = similar_cars['Market_Price(INR)'].mean()
            # Adjust prediction towards market average
            final_prediction = 0.7 * final_prediction + 0.3 * market_avg
        
        st.markdown("---")
        st.subheader("💰 Predicted Market Price")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_price = final_prediction * 0.95
            st.metric("Minimum Price", f"₹{min_price:,.0f}", delta="-5%", delta_color="normal")
        
        with col2:
            st.metric("**Fair Market Price**", f"₹{final_prediction:,.0f}", delta="✓ Recommended", delta_color="off")
        
        with col3:
            max_price = final_prediction * 1.05
            st.metric("Maximum Price", f"₹{max_price:,.0f}", delta="+5%", delta_color="normal")
        
        with col4:
            confidence = (1 - prediction_std/final_prediction) * 100 if final_prediction > 0 else 0
            st.metric("Confidence", f"{min(confidence, 95):.0f}%")
        
        # Price breakdown
        st.markdown("---")
        st.subheader("📊 Price Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Price Components:**")
            if len(similar_cars) > 0:
                st.write(f"• Market Average: ₹{market_avg:,.0f}")
                st.write(f"• Predicted Base: ₹{np.mean(predictions):,.0f}")
                st.write(f"• Final (Adjusted): ₹{final_prediction:,.0f}")
            
            depreciation_rate = 15  # 15% per year
            if 'Year' in inputs:
                age = current_year - inputs['Year']
                depreciation = final_prediction * (depreciation_rate/100) * age
                st.write(f"• Age Depreciation ({age} yrs): ₹{depreciation:,.0f}")
        
        with col2:
            # Price comparison chart
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = ['Min\nPrice', 'Fair\nPrice', 'Max\nPrice']
            values = [min_price, final_prediction, max_price]
            colors = ['#ff6b6b', '#4ecdc4', '#ffe66d']
            ax.bar(labels, values, color=colors, alpha=0.7)
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
            'Predicted_Price': f"₹{final_prediction:,.0f}",
            'Confidence': f"{min(confidence, 95):.0f}%",
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        
        st.success("✅ Price prediction saved to history!")

elif page == "📊 Compare Cars":
    st.subheader("📊 Compare Cars")
    
    num_cars = st.slider("Number of cars", 2, 3, 2)
    
    comparison_data = []
    cols = st.columns(num_cars)
    
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"### Car {i+1}")
            brand = st.selectbox(f"Brand", sorted(df_clean['Brand'].unique()), key=f"b{i}")
            models_list = df_clean[df_clean['Brand'] == brand]['Model'].unique()
            model = st.selectbox(f"Model", sorted(models_list), key=f"m{i}")
            
            car_data = df_clean[(df_clean['Brand'] == brand) & (df_clean['Model'] == model)].iloc[0]
            comparison_data.append({
                'Brand': brand,
                'Model': model,
                'Price': car_data['Market_Price(INR)'],
                'Year': car_data.get('Year', 'N/A')
            })
    
    if st.button("Compare", type="primary"):
        st.markdown("---")
        
        comp_df = pd.DataFrame(comparison_data).T
        comp_df.columns = [f"Car {i+1}" for i in range(num_cars)]
        st.dataframe(comp_df, use_container_width=True)
        
        # Price chart
        fig, ax = plt.subplots(figsize=(10, 6))
        cars = [f"{d['Brand']}\n{d['Model']}" for d in comparison_data]
        prices = [d['Price'] for d in comparison_data]
        ax.bar(cars, prices, color=['#667eea', '#764ba2', '#f093fb'][:num_cars])
        ax.set_ylabel('Price (₹)')
        ax.set_title('Price Comparison')
        plt.ticklabel_format(style='plain', axis='y')
        st.pyplot(fig)
        plt.close()

elif page == "🧮 EMI Calculator":
    st.subheader("🧮 EMI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        price = st.number_input("Car Price (₹)", 100000, 10000000, 1000000, 50000)
        down_payment = st.slider("Down Payment (%)", 0, 50, 20)
        rate = st.slider("Interest Rate (%)", 5.0, 15.0, 9.5, 0.5)
        tenure = st.slider("Tenure (years)", 1, 7, 5)
    
    loan = price * (1 - down_payment/100)
    months = tenure * 12
    monthly_rate = rate / (12 * 100)
    
    if loan > 0:
        emi = loan * monthly_rate * ((1 + monthly_rate)**months) / (((1 + monthly_rate)**months) - 1)
    else:
        emi = 0
    
    total = emi * months
    interest = total - loan
    
    with col2:
        st.markdown("### 💰 EMI Details")
        st.metric("Monthly EMI", f"₹{emi:,.0f}")
        st.metric("Total Amount", f"₹{total:,.0f}")
        st.metric("Total Interest", f"₹{interest:,.0f}")
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie([loan, interest], labels=['Principal', 'Interest'], 
               autopct='%1.1f%%', colors=['#4ecdc4', '#ff6b6b'])
        ax.set_title('Loan Breakdown')
        st.pyplot(fig)
        plt.close()

elif page == "📈 Market Insights":
    st.subheader("📈 Market Insights")
    
    if 'Brand' in df_clean.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top 10 Brands by Count")
            brand_counts_plot = df_clean['Brand'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=brand_counts_plot.values, y=brand_counts_plot.index, palette='viridis')
            ax.set_xlabel('Count')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("### Average Price by Brand")
            brand_avg = df_clean.groupby('Brand')['Market_Price(INR)'].mean().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=brand_avg.values, y=brand_avg.index, palette='rocket')
            ax.set_xlabel('Avg Price (₹)')
            plt.ticklabel_format(style='plain', axis='x')
            st.pyplot(fig)
            plt.close()

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Accurate AI Price Prediction")
