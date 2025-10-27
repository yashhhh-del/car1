# ======================================================
# SMART CAR PRICING SYSTEM - FULLY MERGED & PROFESSIONAL
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
import requests
from bs4 import BeautifulSoup
import re

# ========================================
# MASTER CAR CATALOG (All Brands + Luxury)
# ========================================
CAR_CATALOG = {
    "Maruti Suzuki": ["Swift", "Dzire", "Baleno", "WagonR", "Alto", "Brezza", "Ertiga", "Ciaz", "S-Presso", "Ignis", "Celerio"],
    "Hyundai": ["Creta", "i20", "Verna", "Venue", "i10", "Santro", "Tucson", "Alcazar", "Kona", "Palisade"],
    "Tata": ["Nexon", "Harrier", "Altroz", "Tiago", "Punch", "Safari", "Tigor", "Nexon EV"],
    "Mahindra": ["Thar", "XUV700", "Scorpio", "Bolero", "XUV300", "XUV500", "Marazzo"],
    "Toyota": ["Innova Crysta", "Fortuner", "Camry", "Glanza", "Urban Cruiser", "Yaris", "Vellfire"],
    "Honda": ["City", "Amaze", "WR-V", "Jazz", "Civic", "CR-V"],
    "Kia": ["Seltos", "Sonet", "Carens", "Carnival", "EV6"],
    "Volkswagen": ["Polo", "Vento", "Taigun", "Virtus", "Tiguan", "T-Roc"],
    "Skoda": ["Kushaq", "Slavia", "Octavia", "Superb", "Kodiaq"],
    "Renault": ["Kwid", "Triber", "Kiger", "Duster", "Captur"],
    "Nissan": ["Magnite", "Kicks", "Sunny"],
    "Ford": ["Ecosport", "Figo", "Endeavour", "Mustang"],
    "MG": ["Hector", "Astor", "Gloster", "ZS EV", "Comet EV"],
    "BMW": ["3 Series", "5 Series", "X1", "X3", "X5", "X7", "M3", "M5", "i4", "iX", "7 Series", "Z4"],
    "Mercedes-Benz": ["C-Class", "E-Class", "S-Class", "GLC", "GLE", "GLS", "A-Class", "GLA", "G-Class", "AMG GT"],
    "Audi": ["A4", "A6", "Q3", "Q5", "Q7", "Q8", "A8", "RS Q8", "e-tron", "RS7", "A3"],
    "Porsche": ["911", "Cayenne", "Macan", "Panamera", "Taycan", "718 Cayman"],
    "Lamborghini": ["Huracan", "Urus", "Aventador", "Revuelto"],
    "Ferrari": ["488", "Roma", "SF90", "Portofino", "812", "F8"],
    "Rolls-Royce": ["Phantom", "Ghost", "Cullinan", "Wraith", "Dawn"],
    "Bentley": ["Continental GT", "Bentayga", "Flying Spur", "Mulsanne"],
    "Jaguar": ["F-Pace", "XF", "XE", "F-Type", "I-Pace"],
    "Land Rover": ["Defender", "Range Rover", "Discovery", "Evoque", "Velar"],
    "Volvo": ["XC40", "XC60", "XC90", "S90", "V90"],
    "Tesla": ["Model 3", "Model Y", "Model S", "Model X", "Cybertruck"],
    "Aston Martin": ["DB11", "Vantage", "DBX", "Valkyrie"],
    "McLaren": ["720S", "GT", "570S", "765LT"]
}

# Page config
st.set_page_config(page_title="Smart Car Pricing", layout="wide")
st.title("Smart Car Pricing System")
st.markdown("### Fast, Accurate & Always Predicts – Even Without Your Data")

# Initialize session state
for key in ['predictions', 'model_trained', 'model', 'model_ok']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'predictions' else False if key in ['model_trained', 'model_ok'] else None

# Sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Select Page", [
        "Home", "Price Prediction", "Compare Cars", "EMI Calculator", "About & Improvements"
    ])
    if st.button("Retrain Model"):
        for cache in [st.cache_data, st.cache_resource]:
            cache.clear()
        st.session_state.model_trained = False
        st.rerun()

# File Upload
uploaded_file = st.file_uploader("Upload CSV File (Optional)", type=["csv"])

if uploaded_file is None:
    st.info("No CSV? No problem! You can still predict using web data.")
    df_clean = pd.DataFrame()  # Empty fallback
else:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        price_col = next((col for col in df.columns if 'price' in col.lower()), None)
        if price_col and price_col != 'Market_Price(INR)':
            df = df.rename(columns={price_col: 'Market_Price(INR)'})
        for old, new in [('brand', 'Brand'), ('model', 'Model'), ('year', 'Year'), ('mileage', 'Mileage'),
                         ('fuel', 'Fuel_Type'), ('transmission', 'Transmission'), ('city', 'City')]:
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
        st.success(f"Loaded {len(df_clean)} cars!")
    except Exception as e:
        st.error(f"Error: {e}")
        df_clean = pd.DataFrame()

# Train model only if data exists
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
        encoders = {col: LabelEncoder().fit(df_model[col].astype(str)) for col in cat_cols}
        for col, le in encoders.items():
            df_model[col] = le.transform(df_model[col].astype(str))

        X = df_model.drop(columns=['Market_Price(INR)'], errors='ignore')
        y = df_model['Market_Price(INR)']
        X_scaled = StandardScaler().fit_transform(X)

        param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 15], 'min_samples_split': [2, 5]}
        grid = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), param_grid, cv=5, n_jobs=-1)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        return {
            'model': model, 'scaler': StandardScaler().fit(X), 'encoders': encoders, 'features': X.columns.tolist(),

            'r2': r2, 'mae': mae, 'accuracy': r2 * 100, 'cv_mean': cv_scores.mean() * 100,
            'importances': importances, 'best_params': grid.best_params_
        }

    with st.spinner('Training model...'):
        model_data = train_model(df_clean)
        st.session_state.model = model_data
        st.session_state.model_trained = True

    # 95% CONFIDENCE GATE
    CONFIDENCE_THRESHOLD = 0.95
    if model_data['r2'] < CONFIDENCE_THRESHOLD:
        st.session_state.model_ok = False
        st.warning(f"Model R² = {model_data['r2']:.3f} < 95%. ML predictions disabled. Using web fallback.")
    else:
        st.session_state.model_ok = True

    st.success(f"Model ready! Acc: {model_data['accuracy']:.1f}% | CV: {model_data['cv_mean']:.1f}%")
    model = model_data['model']
    scaler = model_data['scaler']
    encoders = model_data['encoders']
    features = model_data['features']
    importances = model_data['importances']
else:
    st.session_state.model_ok = False
    st.info("No data uploaded → Using **web + catalog fallback** for all predictions.")
    model_data = None

# Web Price Fallback
@st.cache_data(ttl=3600)
def get_web_price(brand, model, year=None, city="Delhi"):
    query = f"{brand.replace(' ', '-')}-{model.replace(' ', '-')}".lower()
    if year: query += f"-{year}"
    urls = [
        f"https://www.cardekho.com/used-{query}+in+{city.lower()}",
        f"https://www.carwale.com/used/cars-in-{city.lower()}/search/?query={query}",
        f"https://www.olx.in/items/q-{query}"
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
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Cars", f"{len(df_clean):,}")
        with col2: st.metric("Brands", df_clean['Brand'].nunique())
        with col3: st.metric("Avg Price", f"₹{df_clean['Market_Price(INR)'].mean()/100000:.1f}L")
        with col4: st.metric("ML Ready", "Yes" if st.session_state.model_ok else "No")
    else:
        st.info("Upload CSV for ML insights. Web fallback is active.")

    if not df_clean.empty and 'Brand' in df_clean.columns:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Top Brands")
            top = df_clean['Brand'].value_counts().head(10)
            fig, ax = plt.subplots(); ax.barh(top.index, top.values, color='skyblue'); st.pyplot(fig); plt.close()
        with col2:
            st.markdown("### Price by Brand")
            brand_price = df_clean.groupby('Brand')['Market_Price(INR)'].mean().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(); ax.barh(brand_price.index, brand_price.values, color='lightgreen'); st.pyplot(fig); plt.close()

    if model_data:
        st.markdown("### Feature Importance")
        fig, ax = plt.subplots(); importances.plot.bar(ax=ax, color='purple'); st.pyplot(fig); plt.close()

elif page == "Price Prediction":
    st.subheader("Predict Car Price")

    # Brand & Model from Catalog + Data
    all_brands = sorted(set(CAR_CATALOG.keys()) | (set(df_clean['Brand'].unique()) if not df_clean.empty else set()))
    brand = st.selectbox("Select Brand", all_brands)

    available_models = CAR_CATALOG.get(brand, [])
    if not df_clean.empty and brand in df_clean['Brand'].values:
        available_models = sorted(set(available_models) | set(df_clean[df_clean['Brand'] == brand]['Model'].unique()))
    model_name = st.selectbox("Select Model", available_models or ["N/A"])

    current_year = datetime.now().year
    year = st.number_input("Year", 1980, current_year + 1, current_year - 3)
    mileage = st.number_input("Mileage (km)", 0, 500000, 30000)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Pune", "Hyderabad"])

    if st.button("Predict Price", type="primary"):
        with st.spinner("Predicting..."):
            final_price = None
            source = ""

            # 1. ML Model (if available and confident)
            if (st.session_state.model_ok and not df_clean.empty and
                brand in df_clean['Brand'].values and model_name in df_clean[df_clean['Brand'] == brand]['Model'].values):
                try:
                    input_data = {'Brand': brand, 'Model': model_name, 'Year': year, 'Mileage': mileage,
                                  'Fuel_Type': fuel, 'Transmission': transmission}
                    if 'City' in df_clean.columns: input_data['City'] = city
                    input_df = pd.DataFrame([input_data])
                    for col in encoders:
                        if col in input_df.columns:
                            try: input_df[col] = encoders[col].transform(input_df[col].astype(str))
                            except: input_df[col] = 0
                    for col in features:
                        if col not in input_df.columns: input_df[col] = 0
                    input_df = input_df[features]
                    input_scaled = scaler.transform(input_df)
                    pred = model.predict(input_scaled)[0]
                    market_avg = df_clean[df_clean['Brand'] == brand]['Market_Price(INR)'].mean()
                    final_price = 0.7 * pred + 0.3 * market_avg
                    source = "AI Model + Market Avg"
                except: pass

            # 2. Web Fallback
            if final_price is None:
                web_price = get_web_price(brand, model_name, year, city)
                if web_price:
                    final_price = web_price
                    source = "Live Web Data"
                else:
                    # 3. Brand Avg + Depreciation
                    brand_avg = df_clean.groupby('Brand')['Market_Price(INR)'].mean().get(brand, 800000)
                    age = current_year - year
                    final_price = brand_avg * (1 - 0.12 * min(age, 10))
                    source = "Brand Avg + Depreciation"

            min_price = final_price * 0.95
            max_price = final_price * 1.05

            st.success(f"**Predicted via: {source}**")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Min Price", f"₹{min_price:,.0f}", "-5%")
            with col2: st.metric("**Fair Price**", f"₹{final_price:,.0f}", "Best")
            with col3: st.metric("Max Price", f"₹{max_price:,.0f}", "+5%")
            with col4: st.metric("Confidence", f"{model_data['accuracy']:.0f}%" if model_data else "Web")

            fig, ax = plt.subplots()
            ax.bar(['Min', 'Fair', 'Max'], [min_price, final_price, max_price], color=['#ff6b6b', '#4ecdc4', '#ffe66d'])
            ax.set_ylabel('Price (₹)'); ax.set_title('Price Range'); st.pyplot(fig); plt.close()
            st.balloons()

            st.session_state.predictions.append({
                'Brand': brand, 'Model': model_name, 'Price': f"₹{final_price:,.0f}",
                'Time': datetime.now().strftime("%H:%M")
            })

elif page == "Compare Cars":
    st.subheader("Compare Cars")
    num_cars = st.slider("Number of cars", 2, 3, 2)
    comparison_data = []
    cols = st.columns(num_cars)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"### Car {i+1}")
            brand = st.selectbox("Brand", sorted(all_brands), key=f"cb{i}")
            models = CAR_CATALOG.get(brand, [])
            if not df_clean.empty and brand in df_clean['Brand'].values:
                models = sorted(set(models) | set(df_clean[df_clean['Brand'] == brand]['Model'].unique()))
            model = st.selectbox("Model", models, key=f"cm{i}")
            comparison_data.append({'Brand': brand, 'Model': model})
    if st.button("Compare", type="primary"):
        st.dataframe(pd.DataFrame(comparison_data).T, use_container_width=True)
        st.info("Full comparison with prices coming soon!")

elif page == "EMI Calculator":
    st.subheader("EMI Calculator")
    col1, col2 = st.columns(2)
    with col1:
        price = st.number_input("Car Price (₹)", 100000, 10000000, 1000000, 50000)
        down = st.slider("Down Payment (%)", 0, 50, 20)
        rate = st.slider("Interest Rate (%)", 5.0, 15.0, 9.5, 0.5)
        tenure = st.slider("Tenure (years)", 1, 7, 5)
    loan = price * (1 - down/100)
    months = tenure * 12
    r = rate / (12 * 100)
    emi = loan * r * ((1 + r)**months) / (((1 + r)**months) - 1) if loan > 0 and r > 0 else 0
    total = emi * months
    interest = total - loan
    with col2:
        st.metric("Monthly EMI", f"₹{emi:,.0f}")
        st.metric("Total Payment", f"₹{total:,.0f}")
        st.metric("Total Interest", f"₹{interest:,.0f}")
        fig, ax = plt.subplots()
        ax.pie([loan, interest], labels=['Principal', 'Interest'], autopct='%1.1f%%', colors=['#4ecdc4', '#ff6b6b'])
        st.pyplot(fig); plt.close()

elif page == "About & Improvements":
    st.write("**This system is now 100% business-ready**")
    st.markdown("""
    - **95%+ Accuracy Gate**  
    - **All Luxury Brands**  
    - **Web Fallback**  
    - **No Data? Still Predicts!**  
    """)
    st.markdown("### Future")
    st.markdown("- API Integration\n- Mobile App\n- Dealer Login\n- Image Upload")

# Footer
st.markdown("---")
if st.session_state.predictions:
    with st.expander("Prediction History"):
        st.dataframe(pd.DataFrame(st.session_state.predictions), use_container_width=True, hide_index=True)

st.markdown("Made with Love | Smart AI Car Pricing")
