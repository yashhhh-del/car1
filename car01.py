# ======================================================
# SMART CAR PRICING SYSTEM - FINAL DEPLOYABLE VERSION
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re

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

# Page config
st.set_page_config(page_title="Smart Car Pricing", layout="wide", initial_sidebar_state="expanded")
st.title("Smart Car Pricing System")
st.markdown("### Tumhara Data Pehle • Web Fallback • 95%+ Accuracy")

# Session state
for key in ['predictions', 'model_trained', 'model', 'model_ok']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'predictions' else False if key in ['model_trained', 'model_ok'] else None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/car.png")
    st.title("Navigation")
    page = st.radio("Go to", ["Home", "Price Prediction", "EMI Calculator", "About"])
    if st.button("Retrain Model"):
        for cache in [st.cache_data, st.cache_resource]:
            cache.clear()
        st.session_state.model_trained = False
        st.rerun()

# File Upload
uploaded_file = st.file_uploader("Upload CSV (Optional)", type=["csv"])

# Load Data
if uploaded_file is None:
    st.info("No CSV? No problem! Web se price laayenge.")
    df_clean = pd.DataFrame()
else:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        price_col = next((c for c in df.columns if 'price' in c.lower()), None)
        if price_col and price_col != 'Market_Price(INR)':
            df = df.rename(columns={price_col: 'Market_Price(INR)'})
        rename_map = {'brand': 'Brand', 'model': 'Model', 'year': 'Year', 'mileage': 'Mileage',
                      'fuel': 'Fuel_Type', 'transmission': 'Transmission', 'city': 'City'}
        for old, new in rename_map.items():
            for col in df.columns:
                if old in col.lower() and col != new:
                    df = df.rename(columns={col: new})
                    break
        df = df.dropna()
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df = df.dropna(subset=['Year']).astype({'Year': int})
        return df
    try:
        df_clean = load_data(uploaded_file)
        st.success(f"Loaded {len(df_clean):,} cars!")
    except Exception as e:
        st.error(f"Error: {e}")
        df_clean = pd.DataFrame()

# Train Model
if not df_clean.empty and 'Market_Price(INR)' in df_clean.columns:
    @st.cache_resource
    def train_model(df):
        current_year = datetime.now().year
        df_model = df.copy()
        if 'Year' in df_model.columns:
            df_model['Car_Age'] = current_year - df_model['Year']
        if 'Brand' in df_model.columns:
            df_model['Brand_Avg_Price'] = df_model['Brand'].map(df_model.groupby('Brand')['Market_Price(INR)'].mean())

        cat_cols = df_model.select_dtypes(include=['object']).columns
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            encoders[col] = le

        X = df_model.drop(columns=['Market_Price(INR)'], errors='ignore')
        y = df_model['Market_Price(INR)']
        X_scaled = StandardScaler().fit_transform(X)

        param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 15]}
        grid = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), param_grid, cv=5)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        return {
            'model': model, 'scaler': StandardScaler().fit(X), 'encoders': encoders, 'features': X.columns.tolist(),
            'r2': r2, 'accuracy': r2 * 100, 'cv_mean': cv_scores.mean() * 100, 'importances': importances
        }

    with st.spinner('Training AI model...'):
        model_data = train_model(df_clean)
        st.session_state.model = model_data
        st.session_state.model_trained = True

    if model_data['r2'] < 0.95:
        st.session_state.model_ok = False
        st.warning("AI Model < 95% accuracy → Using Web Fallback")
    else:
        st.session_state.model_ok = True
        st.success(f"AI Model Ready! Accuracy: {model_data['accuracy']:.1f}%")

    model = model_data['model']
    scaler = model_data['scaler']
    encoders = model_data['encoders']
    features = model_data['features']
    importances = model_data['importances']
else:
    st.session_state.model_ok = False
    st.info("No data → Web + Catalog se prediction")

# Web Fallback
@st.cache_data(ttl=3600)
def get_web_price(brand, model, year=None, city="Delhi"):
    query = f"{brand.replace(' ', '-')}-{model.replace(' ', '-')}".lower()
    if year: query += f"-{year}"
    urls = [
        f"https://www.cardekho.com/used-{query}+in+{city.lower()}",
        f"https://www.carwale.com/used/cars-in-{city.lower()}/search/?query={query}"
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    prices = []
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code != 200: continue
            soup = BeautifulSoup(r.text, 'html.parser')
            texts = soup.find_all(string=re.compile(r'₹'))
            for t in texts:
                m = re.search(r'₹\s*([\d,.]+)\s*(lakh|crore)?', t, re.I)
                if m:
                    val = float(m.group(1).replace(',', ''))
                    if m.group(2) and 'crore' in m.group(2).lower(): val *= 100
                    prices.append(int(val * 100000))
        except: continue
    return int(np.mean(prices)) if prices else None

# ============================================
# PAGES
# ============================================

if page == "Home":
    st.subheader("Market Overview")
    if not df_clean.empty:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Cars", f"{len(df_clean):,}")
        with col2: st.metric("Brands", df_clean['Brand'].nunique())
        with col3: st.metric("Avg Price", f"₹{df_clean['Market_Price(INR)'].mean()/100000:.1f}L")

        # Interactive Brand Price
        brand_price = df_clean.groupby('Brand')['Market_Price(INR)'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(y=brand_price.index, x=brand_price.values, orientation='h',
                     text=[f"₹{v/100000:.1f}LL" for v in brand_price.values],
                     color_discrete_sequence=['#4ecdc4'])
        fig.update_layout(title="Top 10 Brands by Avg Price", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Feature Importance
        if 'importances' in locals():
            fig = px.bar(y=importances.index, x=importances.values, orientation='h',
                         text=[f"{v:.3f}" for v in importances.values],
                         color_discrete_sequence=['#9b59b6'])
            fig.update_layout(title="What Affects Price Most?", height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload CSV to see insights")

elif page == "Price Prediction":
    st.subheader("Predict Car Price")

    # DATA FIRST
    data_brands = sorted(df_clean['Brand'].unique().tolist()) if not df_clean.empty and 'Brand' in df_clean.columns else []
    catalog_brands = list(CAR_CATALOG.keys())
    all_brands = data_brands + [b for b in catalog_brands if b not in data_brands]
    brand = st.selectbox("Select Brand", all_brands)

    available_models = []
    if not df_clean.empty and brand in df_clean['Brand'].values:
        available_models = sorted(df_clean[df_clean['Brand'] == brand]['Model'].unique().tolist())
    if not available_models and brand in CAR_CATALOG:
        available_models = CAR_CATALOG[brand]

    model_name = st.selectbox("Select Model", available_models or ["No models"])
    if not available_models:
        st.warning("No model found")
        st.stop()

    if brand in data_brands:
        st.success(f"{brand} → From your CSV!")
    else:
        st.info(f"{brand} → From catalog")

    current_year = datetime.now().year
    year = st.number_input("Year", 1980, current_year + 1, current_year - 3)
    mileage = st.number_input("Mileage (km)", 0, 500000, 30000)
    fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune"])

    if st.button("Predict Price", type="primary"):
        with st.spinner("Calculating..."):
            final_price = None
            source = ""

            if (st.session_state.model_ok and not df_clean.empty and
                brand in df_clean['Brand'].values and model_name in df_clean[df_clean['Brand'] == brand ]['Model'].values):
                try:
                    input_data = {'Brand': brand, 'Model': model_name, 'Year': year, 'Mileage': mileage,
                                  'Fuel_Type': fuel, 'Transmission': transmission}
                    if 'City' in df_clean.columns: input_data['City'] = city
                    input_df = pd.DataFrame([input_data])
                    for col in encoders:
                        if col in input_df.columns:
                            try: input_df[col] = encoders[col].transform([input_data[col]])[0]
                            except: input_df[col] = 0
                    for col in features:
                        if col not in input_df.columns: input_df[col] = 0
                    input_df = input_df[features]
                    input_scaled = scaler.transform(input_df)
                    pred = model.predict(input_scaled)[0]
                    market_avg = df_clean[df_clean['Brand'] == brand]['Market_Price(INR)'].mean()
                    final_price = 0.7 * pred + 0.3 * market_avg
                    source = "AI Model"
                except: pass

            if final_price is None:
                web_price = get_web_price(brand, model_name, year, city)
                if web_price:
                    final_price = web_price
                    source = "Live Web"
                else:
                    brand_avg = df_clean['Market_Price(INR)'].mean() if not df_clean.empty else 800000
                    age = current_year - year
                    final_price = brand_avg * (1 - 0.12 * min(age, 10))
                    source = "Estimate"

            min_price = final_price * 0.95
            max_price = final_price * 1.05

            st.success(f"Predicted via: {source}")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Min', 'Fair', 'Max'], y=[min_price, final_price, max_price],
                marker_color=['#ff6b6b', '#1a936f', '#ffe66d'],
                text=[f"₹{v:,.0f}" for v in [min_price, final_price, max_price]],
                textposition='outside'
            ))
            fig.update_layout(title="Price Range", height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.balloons()

            st.session_state.predictions.append({
                'Brand': brand, 'Model': model_name, 'Price': f"₹{final_price:,.0f}",
                'Time': datetime.now().strftime("%H:%M")
            })

elif page == "EMI Calculator":
    st.subheader("EMI Calculator")
    price = st.number_input("Car Price", 100000, 10000000, 1000000, 50000)
    down = st.slider("Down Payment (%)", 0, 50, 20)
    rate = st.slider("Interest Rate (%)", 5.0, 15.0, 9.5, 0.1)
    tenure = st.slider("Tenure (years)", 1, 7, 5)
    loan = price * (1 - down/100)
    r = rate / (12 * 100)
    months = tenure * 12
    emi = loan * r * ((1 + r)**months) / (((1 + r)**months) - 1) if loan > 0 else 0
    total = emi * months
    interest = total - loan

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Monthly EMI", f"₹{emi:,.0f}")
        st.metric("Total Interest", f"₹{interest:,.0f}")
    with col2:
        fig = go.Figure(data=[go.Pie(labels=['Principal', 'Interest'], values=[loan, interest],
                                    hole=0.4, marker_colors=['#4ecdc4', '#ff6b6b'])])
        fig.update_layout(title="EMI Breakdown")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
if st.session_state.predictions:
    with st.expander("Prediction History"):
        hist_df = pd.DataFrame(st.session_state.predictions)
        st.dataframe(hist_df, use_container_width=True)

st.markdown("Made with Love | Tumhara App Ab Live Hai!")

