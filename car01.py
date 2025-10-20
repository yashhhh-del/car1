# ======================================================
# SMART CAR PRICING SYSTEM - FIXED VERSION
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Page config
st.set_page_config(page_title="Smart Car Pricing", layout="wide")

# Title
st.title("🚗 Smart Car Pricing System")
st.markdown("### AI-Powered Price Prediction")

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
        "🧮 EMI Calculator"
    ])

# File Upload
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.info("👆 Please upload a CSV file to get started!")
    st.markdown("---")
    st.markdown("### 📋 Required Columns:")
    st.write("- Brand")
    st.write("- Model")
    st.write("- Market_Price(INR)")
    st.write("- Year")
    st.stop()

# Load data
try:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading file: {e}")
    st.stop()

# Show uploaded columns
with st.expander("📋 View Uploaded Columns"):
    st.write("**Columns in your file:**")
    for i, col in enumerate(df.columns, 1):
        st.write(f"{i}. {col}")

# Find price column (auto-detect)
price_col = None
possible_names = ['Market_Price(INR)', 'Market_Price', 'Price', 'price', 'PRICE', 
                  'Market Price', 'market_price', 'Price(INR)', 'Selling_Price',
                  'selling_price', 'Car_Price', 'car_price']

for col in df.columns:
    if col in possible_names or 'price' in col.lower():
        price_col = col
        break

if price_col is None:
    st.error("❌ Price column not found!")
    st.info("💡 Please ensure your CSV has a column with 'Price' in its name")
    st.write("**Available columns:**", list(df.columns))
    st.markdown("---")
    st.markdown("### Sample CSV Format:")
    st.code("""Brand,Model,Year,Price,Fuel_Type
Maruti,Swift,2020,550000,Petrol
Honda,City,2019,900000,Petrol""")
    st.stop()

# Rename to standard name
if price_col != 'Market_Price(INR)':
    df = df.rename(columns={price_col: 'Market_Price(INR)'})
    st.info(f"✅ Using '{price_col}' as price column")

# Find Brand column
brand_col = None
for col in df.columns:
    if 'brand' in col.lower():
        brand_col = col
        break

if brand_col and brand_col != 'Brand':
    df = df.rename(columns={brand_col: 'Brand'})

# Find Model column
model_col = None
for col in df.columns:
    if 'model' in col.lower():
        model_col = col
        break

if model_col and model_col != 'Model':
    df = df.rename(columns={model_col: 'Model'})

# Check essential columns
if 'Brand' not in df.columns or 'Model' not in df.columns:
    st.warning("⚠️ Brand or Model column not found. Some features may not work.")

# Clean data
df_clean = df.dropna()

if 'Year' in df_clean.columns:
    df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Year'])
    df_clean['Year'] = df_clean['Year'].astype(int)

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

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

with st.spinner('Training model...'):
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

st.success(f"✅ Model trained! Accuracy: {r2:.2%}")

# ============================================
# PAGES
# ============================================

if page == "🏠 Home":
    st.subheader("📊 Market Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cars", f"{len(df_clean):,}")
    with col2:
        st.metric("Unique Brands", f"{df_clean['Brand'].nunique()}")
    with col3:
        avg_price = df_clean['Market_Price(INR)'].mean()
        st.metric("Avg. Price", f"₹{avg_price/100000:.1f}L")
    
    st.markdown("---")
    
    # Show data sample
    st.subheader("📋 Data Sample")
    st.dataframe(df_clean.head(10), use_container_width=True)
    
    # Price distribution
    if 'Brand' in df_clean.columns:
        st.subheader("📈 Price by Brand")
        top_brands = df_clean['Brand'].value_counts().head(5).index
        filtered = df_clean[df_clean['Brand'].isin(top_brands)]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=filtered, x='Brand', y='Market_Price(INR)', ax=ax)
        ax.set_title('Price Distribution by Top 5 Brands')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()

elif page == "💰 Price Prediction":
    st.subheader("💰 Predict Car Price")
    
    if 'Brand' not in df_clean.columns or 'Model' not in df_clean.columns:
        st.error("❌ Brand and Model columns required!")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox("Select Brand", sorted(df_clean['Brand'].unique()))
    
    with col2:
        models = df_clean[df_clean['Brand'] == brand]['Model'].unique()
        model_name = st.selectbox("Select Model", sorted(models))
    
    # Get sample data
    sample = df_clean[(df_clean['Brand'] == brand) & (df_clean['Model'] == model_name)].iloc[0]
    
    # Input features
    st.markdown("### 📝 Car Details")
    
    inputs = {}
    for col in X.columns:
        if col in sample.index:
            if col in encoders:
                # Categorical
                options = sorted(df_clean[col].unique())
                default = sample[col] if sample[col] in options else options[0]
                inputs[col] = st.selectbox(f"{col}", options, index=options.index(default))
            else:
                # Numerical
                min_val = int(df_clean[col].min())
                max_val = int(df_clean[col].max())
                default = int(sample[col])
                inputs[col] = st.slider(f"{col}", min_val, max_val, default)
    
    if st.button("🔍 Predict Price", type="primary"):
        # Prepare input
        input_data = {}
        for col in X.columns:
            if col in inputs:
                input_data[col] = inputs[col]
            else:
                input_data[col] = sample[col] if col in sample.index else 0
        
        input_df = pd.DataFrame([input_data])
        
        # Encode categoricals
        for col in encoders:
            if col in input_df.columns:
                try:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str))
                except:
                    input_df[col] = encoders[col].transform([encoders[col].classes_[0]])
        
        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        
        st.markdown("---")
        st.subheader("💰 Predicted Price")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Price", f"₹{prediction*0.9:,.0f}")
        with col2:
            st.metric("Fair Price", f"₹{prediction:,.0f}")
        with col3:
            st.metric("Max Price", f"₹{prediction*1.1:,.0f}")
        
        st.balloons()

elif page == "📊 Compare Cars":
    st.subheader("📊 Compare Two Cars")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Car 1")
        brand1 = st.selectbox("Brand", sorted(df_clean['Brand'].unique()), key="b1")
        models1 = df_clean[df_clean['Brand'] == brand1]['Model'].unique()
        model1 = st.selectbox("Model", sorted(models1), key="m1")
        car1 = df_clean[(df_clean['Brand'] == brand1) & (df_clean['Model'] == model1)].iloc[0]
    
    with col2:
        st.markdown("### Car 2")
        brand2 = st.selectbox("Brand", sorted(df_clean['Brand'].unique()), key="b2")
        models2 = df_clean[df_clean['Brand'] == brand2]['Model'].unique()
        model2 = st.selectbox("Model", sorted(models2), key="m2")
        car2 = df_clean[(df_clean['Brand'] == brand2) & (df_clean['Model'] == model2)].iloc[0]
    
    if st.button("Compare", type="primary"):
        st.markdown("---")
        
        comparison = {
            'Feature': ['Brand', 'Model', 'Price', 'Year'],
            'Car 1': [brand1, model1, f"₹{car1['Market_Price(INR)']:,.0f}", car1['Year']],
            'Car 2': [brand2, model2, f"₹{car2['Market_Price(INR)']:,.0f}", car2['Year']]
        }
        
        st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
        
        # Price chart
        fig, ax = plt.subplots(figsize=(8, 5))
        cars = [f"{brand1}\n{model1}", f"{brand2}\n{model2}"]
        prices = [car1['Market_Price(INR)'], car2['Market_Price(INR)']]
        ax.bar(cars, prices, color=['#667eea', '#764ba2'])
        ax.set_ylabel('Price (INR)')
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
    
    with col2:
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
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Monthly EMI", f"₹{emi:,.0f}")
    with col2:
        st.metric("Loan Amount", f"₹{loan:,.0f}")
    with col3:
        st.metric("Total Interest", f"₹{interest:,.0f}")
    with col4:
        st.metric("Total Payment", f"₹{total:,.0f}")
    
    # Pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([loan, interest], labels=['Principal', 'Interest'], 
           autopct='%1.1f%%', colors=['#667eea', '#764ba2'])
    ax.set_title('Payment Breakdown')
    st.pyplot(fig)
    plt.close()

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
