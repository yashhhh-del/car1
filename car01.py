import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

st.set_page_config(page_title="Smart Car Pricing", layout="wide")
st.title("🚗 Smart Car Pricing System")
st.markdown("### Accurate AI Price Prediction from Your CSV Data")

if 'predictions' not in st.session_state:
    st.session_state.predictions = []

with st.sidebar:
    st.title("📊 Navigation")
    page = st.radio("Select Page", ["🏠 Home", "💰 Price Prediction", "📊 Compare Cars", "🧮 EMI Calculator"])

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.info("👆 Upload CSV file to start!")
    st.code("""Brand,Model,Year,Mileage,Fuel_Type,Transmission,Price
Maruti,Swift,2020,15000,Petrol,Manual,550000""")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    for col in df.columns:
        if 'price' in col.lower():
            df = df.rename(columns={col: 'Market_Price(INR)'})
            break
    for old, new in [('brand', 'Brand'), ('model', 'Model'), ('year', 'Year')]:
        for col in df.columns:
            if old in col.lower() and col != new:
                df = df.rename(columns={col: new})
                break
    df = df.drop_duplicates().dropna()
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year'])
        df['Year'] = df['Year'].astype(int)
        df = df[(df['Year'] >= 1980) & (df['Year'] <= datetime.now().year)]
    if 'Market_Price(INR)' in df.columns:
        Q1 = df['Market_Price(INR)'].quantile(0.01)
        Q3 = df['Market_Price(INR)'].quantile(0.99)
        df = df[(df['Market_Price(INR)'] >= Q1) & (df['Market_Price(INR)'] <= Q3)]
    return df

try:
    df_clean = load_data(uploaded_file)
    st.success(f"✅ Loaded {len(df_clean)} cars")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

if 'Market_Price(INR)' not in df_clean.columns or 'Brand' not in df_clean.columns or 'Model' not in df_clean.columns:
    st.error("❌ Required columns missing!")
    st.stop()

@st.cache_resource
def train_model(df):
    current_year = datetime.now().year
    df_model = df.copy()
    if 'Year' in df_model.columns:
        df_model['Car_Age'] = current_year - df_model['Year']
    if 'Brand' in df_model.columns:
        df_model['Brand_Avg_Price'] = df_model['Brand'].map(df_model.groupby('Brand')['Market_Price(INR)'].mean())
    encoders = {}
    for col in df_model.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le
    X = df_model.drop(columns=['Market_Price(INR)'], errors='ignore')
    y = df_model['Market_Price(INR)']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return {'model': model, 'scaler': scaler, 'encoders': encoders, 'features': X.columns.tolist(), 'accuracy': r2 * 100}

with st.spinner('🎯 Training model...'):
    model_data = train_model(df_clean)

st.metric("Model Accuracy", f"{model_data['accuracy']:.1f}%")

if page == "🏠 Home":
    st.subheader("📊 Market Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cars", f"{len(df_clean):,}")
    col2.metric("Brands", f"{df_clean['Brand'].nunique()}")
    col3.metric("Avg Price", f"₹{df_clean['Market_Price(INR)'].mean()/100000:.1f}L")
    st.markdown("---")
    st.markdown("### 📈 Top 10 Brands")
    top_brands = df_clean['Brand'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_brands.index, top_brands.values, color='skyblue')
    ax.set_xlabel('Count')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown("---")
    st.dataframe(df_clean.head(20), use_container_width=True)

elif page == "💰 Price Prediction":
    st.subheader("💰 Get Accurate Car Price")
    brand = st.selectbox("🚘 Brand", sorted(df_clean['Brand'].unique()))
    brand_data = df_clean[df_clean['Brand'] == brand]
    model_name = st.selectbox("🔧 Model", sorted(brand_data['Model'].unique()))
    selected_car_data = brand_data[brand_data['Model'] == model_name]
    
    if len(selected_car_data) == 0:
        st.warning("No data found")
        st.stop()
    
    avg_price = selected_car_data['Market_Price(INR)'].mean()
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Cars Found", len(selected_car_data))
    col2.metric("Avg Price", f"₹{avg_price:,.0f}")
    col3.metric("Range", f"₹{selected_car_data['Market_Price(INR)'].min():,.0f}-₹{selected_car_data['Market_Price(INR)'].max():,.0f}")
    
    st.markdown("---")
    available_cols = [col for col in selected_car_data.columns if col not in ['Market_Price(INR)', 'Brand', 'Model']]
    
    col1, col2, col3 = st.columns(3)
    inputs = {'Brand': brand, 'Model': model_name}
    col_idx = 0
    
    for col in available_cols:
        with [col1, col2, col3][col_idx % 3]:
            if selected_car_data[col].dtype in ['int64', 'float64']:
                inputs[col] = st.number_input(f"{col}", float(selected_car_data[col].min()), float(selected_car_data[col].max()), float(selected_car_data[col].median()), key=f"inp_{col}")
            else:
                inputs[col] = st.selectbox(f"{col}", sorted(selected_car_data[col].unique()), key=f"inp_{col}")
            col_idx += 1
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        condition = st.select_slider("Condition", ["Poor", "Fair", "Good", "Excellent"], value="Good")
    with col2:
        accident = st.radio("Accident", ["No", "Minor", "Major"], index=0)
    with col3:
        owners = st.number_input("Owners", 1, 5, 1)
    
    if st.button("🔍 Calculate Price", type="primary", use_container_width=True):
        query_df = selected_car_data.copy()
        for col, val in inputs.items():
            if col in query_df.columns and col not in ['Brand', 'Model']:
                if query_df[col].dtype in ['int64', 'float64']:
                    query_df = query_df[(query_df[col] >= val * 0.8) & (query_df[col] <= val * 1.2)]
                else:
                    query_df = query_df[query_df[col] == val]
        
        if len(query_df) >= 2:
            base_price = query_df['Market_Price(INR)'].median()
            similar_count = len(query_df)
        else:
            base_price = selected_car_data['Market_Price(INR)'].median()
            similar_count = len(selected_car_data)
        
        condition_mult = {"Poor": 0.85, "Fair": 0.93, "Good": 1.0, "Excellent": 1.08}
        accident_mult = {"No": 1.0, "Minor": 0.95, "Major": 0.85}
        adjusted_price = base_price * condition_mult[condition] * accident_mult[accident] * (1 - (owners - 1) * 0.03)
        lower_bound = adjusted_price * 0.95
        upper_bound = adjusted_price * 1.05
        
        st.markdown("---")
        st.success("✅ Price Calculated!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Similar Cars", similar_count)
        col2.metric("Base Price", f"₹{base_price:,.0f}")
        col3.metric("**Fair Price**", f"₹{adjusted_price:,.0f}")
        col4.metric("Range", f"±₹{(upper_bound-lower_bound)/2:,.0f}")
        
        st.markdown("---")
        if len(query_df) > 0:
            st.markdown("### 🚗 Similar Cars")
            display_df = query_df[['Brand', 'Model'] + available_cols[:3] + ['Market_Price(INR)']].head(10)
            st.dataframe(display_df.style.format({'Market_Price(INR)': '₹{:,.0f}'}), use_container_width=True)
        
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(['Lower', 'Fair', 'Upper'], [lower_bound, adjusted_price, upper_bound], color=['#ff6b6b', '#4ecdc4', '#ffe66d'], alpha=0.8)
        for bar, val in zip(bars, [lower_bound, adjusted_price, upper_bound]):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'₹{val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel('Price (₹)')
        ax.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.balloons()
        
        st.session_state.predictions.append({'Brand': brand, 'Model': model_name, 'Condition': condition, 'Price': f"₹{adjusted_price:,.0f}", 'Time': datetime.now().strftime("%Y-%m-%d %H:%M")})

elif page == "📊 Compare Cars":
    st.subheader("📊 Compare Cars")
    num_cars = st.slider("Number of cars", 2, 4, 2)
    comparison_data = []
    cols = st.columns(num_cars)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"### Car {i+1}")
            brand = st.selectbox("Brand", sorted(df_clean['Brand'].unique()), key=f"cb{i}")
            model_car = st.selectbox("Model", sorted(df_clean[df_clean['Brand'] == brand]['Model'].unique()), key=f"cm{i}")
            car_data = df_clean[(df_clean['Brand'] == brand) & (df_clean['Model'] == model_car)]
            if len(car_data) > 0:
                comparison_data.append({'Brand': brand, 'Model': model_car, 'Price': car_data['Market_Price(INR)'].mean()})
    
    if st.button("Compare", type="primary"):
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar([f"{d['Brand']}\n{d['Model']}" for d in comparison_data], [d['Price'] for d in comparison_data], color=['#667eea', '#764ba2', '#f093fb', '#fccb90'][:num_cars], alpha=0.8)
        for bar, d in zip(bars, comparison_data):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f"₹{d['Price']:,.0f}", ha='center', va='bottom')
        ax.set_ylabel('Price (₹)')
        ax.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.success(f"💰 Best Value: {comparison_data[0]['Brand']} {comparison_data[0]['Model']}")

elif page == "🧮 EMI Calculator":
    st.subheader("🧮 EMI Calculator")
    col1, col2 = st.columns(2)
    with col1:
        price = st.number_input("Car Price (₹)", 100000, 10000000, 1000000, 50000)
        down = st.slider("Down Payment (%)", 0, 50, 20)
        rate = st.slider("Interest Rate (%)", 5.0, 15.0, 9.5, 0.5)
        tenure = st.slider("Tenure (years)", 1, 7, 5)
    loan = price * (1 - down/100)
    months = tenure * 12
    r = rate / (12 * 100)
    emi = loan * r * ((1 + r)**months) / (((1 + r)**months) - 1) if loan > 0 and r > 0 else (loan / months if months > 0 else 0)
    total = emi * months
    interest = total - loan
    with col2:
        st.markdown("### 💳 EMI Summary")
        st.metric("Monthly EMI", f"₹{emi:,.0f}")
        st.metric("Total Payment", f"₹{total:,.0f}")
        st.metric("Total Interest", f"₹{interest:,.0f}")

st.markdown("---")
if len(st.session_state.predictions) > 0:
    with st.expander("📜 History"):
        st.dataframe(pd.DataFrame(st.session_state.predictions), use_container_width=True)
st.markdown("Made with ❤️ | Smart Car Pricing")
