import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="CarWale - Complete Car Portal",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102,126,234,0.4);
    }
    .car-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .price-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        border: 3px solid #667eea;
    }
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stMetric {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Car Database
CAR_DATABASE = {
    "Mercedes-Benz": {
        "models": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLC", "GLE", "GLS", "AMG GT", "EQC", "Maybach S-Class"],
        "price_range": (4000000, 30000000),
        "category": "Luxury"
    },
    "BMW": {
        "models": ["1 Series", "2 Series", "3 Series", "5 Series", "7 Series", "X1", "X3", "X5", "X7", "Z4", "i4", "iX", "M3", "M5"],
        "price_range": (4200000, 25000000),
        "category": "Luxury"
    },
    "Audi": {
        "models": ["A3", "A4", "A6", "A8", "Q2", "Q3", "Q5", "Q7", "Q8", "e-tron", "RS5", "RS7"],
        "price_range": (3800000, 22000000),
        "category": "Luxury"
    },
    "Tesla": {
        "models": ["Model 3", "Model Y", "Model S", "Model X"],
        "price_range": (6000000, 18000000),
        "category": "Electric Luxury"
    },
    "Porsche": {
        "models": ["718", "911", "Panamera", "Cayenne", "Macan", "Taycan"],
        "price_range": (8000000, 50000000),
        "category": "Super Luxury"
    },
    "Ferrari": {
        "models": ["Roma", "Portofino", "F8 Tributo", "SF90", "812 Superfast", "Purosangue"],
        "price_range": (40000000, 120000000),
        "category": "Super Sports"
    },
    "Lamborghini": {
        "models": ["Huracan", "Urus", "Aventador", "Revuelto"],
        "price_range": (35000000, 100000000),
        "category": "Super Sports"
    },
    "Rolls-Royce": {
        "models": ["Ghost", "Phantom", "Wraith", "Dawn", "Cullinan"],
        "price_range": (50000000, 120000000),
        "category": "Ultra Luxury"
    },
    "Bentley": {
        "models": ["Continental GT", "Flying Spur", "Bentayga", "Mulsanne"],
        "price_range": (30000000, 80000000),
        "category": "Ultra Luxury"
    },
    "Jaguar": {
        "models": ["XE", "XF", "XJ", "F-Type", "E-Pace", "F-Pace", "I-Pace"],
        "price_range": (4700000, 20000000),
        "category": "Luxury"
    },
    "Land Rover": {
        "models": ["Discovery", "Discovery Sport", "Range Rover Evoque", "Range Rover Velar", "Range Rover Sport", "Range Rover", "Defender"],
        "price_range": (6000000, 40000000),
        "category": "Luxury SUV"
    },
    "Toyota": {
        "models": ["Glanza", "Urban Cruiser", "Fortuner", "Innova Crysta", "Camry", "Vellfire", "Hilux", "Land Cruiser"],
        "price_range": (700000, 22000000),
        "category": "Premium"
    },
    "Honda": {
        "models": ["Amaze", "City", "Elevate", "CR-V", "Civic", "Accord"],
        "price_range": (700000, 4500000),
        "category": "Premium"
    },
    "Hyundai": {
        "models": ["Grand i10 Nios", "i20", "Aura", "Verna", "Creta", "Alcazar", "Tucson", "Venue", "Exter", "Ioniq 5"],
        "price_range": (550000, 4500000),
        "category": "Mass Premium"
    },
    "Kia": {
        "models": ["Sonet", "Seltos", "Carens", "EV6", "Carnival"],
        "price_range": (750000, 6500000),
        "category": "Mass Premium"
    },
    "Maruti Suzuki": {
        "models": ["Alto", "S-Presso", "WagonR", "Swift", "Dzire", "Baleno", "Celerio", "Ignis", "Brezza", "Ertiga", "Ciaz", "XL6", "Grand Vitara", "Jimny", "Fronx", "Invicto"],
        "price_range": (350000, 2800000),
        "category": "Mass Market"
    },
    "Tata": {
        "models": ["Tiago", "Tigor", "Altroz", "Punch", "Nexon", "Harrier", "Safari", "Curvv"],
        "price_range": (500000, 2800000),
        "category": "Mass Market"
    },
    "Mahindra": {
        "models": ["Bolero", "Thar", "Scorpio", "Scorpio-N", "XUV300", "XUV700", "XUV400", "Marazzo", "Alturas G4"],
        "price_range": (900000, 2700000),
        "category": "SUV"
    },
    "Volkswagen": {
        "models": ["Polo", "Virtus", "Taigun", "Tiguan"],
        "price_range": (600000, 3500000),
        "category": "Premium"
    },
    "Skoda": {
        "models": ["Kushaq", "Slavia", "Kodiaq", "Superb"],
        "price_range": (1100000, 4000000),
        "category": "Premium"
    },
    "MG": {
        "models": ["Hector", "Astor", "ZS EV", "Gloster", "Comet EV"],
        "price_range": (1000000, 4500000),
        "category": "Mass Premium"
    },
    "Nissan": {
        "models": ["Magnite", "X-Trail", "GT-R"],
        "price_range": (600000, 22000000),
        "category": "Mass Premium"
    },
    "Renault": {
        "models": ["Kwid", "Triber", "Kiger", "Duster"],
        "price_range": (450000, 1500000),
        "category": "Mass Market"
    },
    "Ford": {
        "models": ["EcoSport", "Endeavour", "Mustang"],
        "price_range": (900000, 7500000),
        "category": "Premium"
    },
    "Jeep": {
        "models": ["Compass", "Meridian", "Wrangler", "Grand Cherokee"],
        "price_range": (1800000, 8000000),
        "category": "Premium SUV"
    },
    "BYD": {
        "models": ["Atto 3", "e6", "Seal"],
        "price_range": (2500000, 5500000),
        "category": "Electric"
    },
    "Citroen": {
        "models": ["C3", "C3 Aircross", "eC3"],
        "price_range": (600000, 1500000),
        "category": "Mass Market"
    }
}

# Initialize session state
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None
    st.session_state.encoders = {}
    st.session_state.scaler = None
    st.session_state.feature_columns = []
    st.session_state.predictions_history = []

def format_price(price):
    """Format price in Indian format"""
    if price >= 10000000:
        return f"₹{price/10000000:.2f} Cr"
    elif price >= 100000:
        return f"₹{price/100000:.2f} Lakh"
    else:
        return f"₹{price:,.0f}"

def generate_training_data(num_samples=5000):
    """Generate synthetic training data"""
    data = []
    current_year = datetime.now().year
    
    fuel_types = ["Petrol", "Diesel", "Electric", "Hybrid", "CNG"]
    transmissions = ["Manual", "Automatic", "CVT", "DCT", "AMT"]
    colors = ["White", "Black", "Silver", "Red", "Blue", "Grey", "Brown"]
    conditions = ["Excellent", "Good", "Fair", "Poor"]
    accidents = ["No", "Minor", "Yes"]
    
    for _ in range(num_samples):
        brand = random.choice(list(CAR_DATABASE.keys()))
        brand_info = CAR_DATABASE[brand]
        model = random.choice(brand_info["models"])
        price_min, price_max = brand_info["price_range"]
        
        if random.random() < 0.7:
            year = random.randint(current_year - 5, current_year)
        else:
            year = random.randint(current_year - 15, current_year - 6)
        
        age = current_year - year
        base_price = random.randint(price_min, price_max)
        
        if age == 0:
            depreciation = 1.0
        elif age == 1:
            depreciation = 0.85
        elif age == 2:
            depreciation = 0.75
        else:
            depreciation = max(0.3, 0.75 - (age - 2) * 0.08)
        
        condition = random.choice(conditions)
        condition_factor = {"Excellent": 1.05, "Good": 1.0, "Fair": 0.90, "Poor": 0.75}[condition]
        
        owners = random.choice([1, 1, 1, 2, 2, 3])
        owner_factor = {1: 1.0, 2: 0.95, 3: 0.85, 4: 0.75}[owners]
        
        accident = random.choice(accidents)
        accident_factor = {"No": 1.0, "Minor": 0.92, "Yes": 0.80}[accident]
        
        price = int(base_price * depreciation * condition_factor * owner_factor * accident_factor)
        
        avg_km_per_year = random.randint(8000, 15000)
        mileage = age * avg_km_per_year + random.randint(-2000, 2000)
        mileage = max(0, mileage)
        
        if brand_info["category"] in ["Electric", "Electric Luxury"]:
            fuel = "Electric"
        else:
            fuel = random.choice(fuel_types)
        
        if brand_info["category"] in ["Luxury", "Super Luxury", "Ultra Luxury", "Super Sports"]:
            transmission = random.choice(["Automatic", "DCT"])
        else:
            transmission = random.choice(transmissions)
        
        data.append({
            "Brand": brand,
            "Model": model,
            "Year": year,
            "Mileage": mileage,
            "Fuel_Type": fuel,
            "Transmission": transmission,
            "Color": random.choice(colors),
            "Owners": owners,
            "Condition": condition,
            "Accident_History": accident,
            "Category": brand_info["category"],
            "Price": price
        })
    
    return pd.DataFrame(data)

@st.cache_resource
def train_ml_model():
    """Train the ML model"""
    with st.spinner("🎯 Training AI model... Please wait..."):
        df = generate_training_data(5000)
        
        current_year = datetime.now().year
        df['Car_Age'] = current_year - df['Year']
        df['Brand_Avg_Price'] = df['Brand'].map(df.groupby('Brand')['Price'].mean())
        
        encoders = {}
        categorical_cols = ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Color', 'Condition', 'Accident_History', 'Category']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        
        feature_columns = ['Year', 'Mileage', 'Owners', 'Car_Age', 'Brand_Avg_Price'] + \
                         [col + '_Encoded' for col in categorical_cols]
        
        X = df[feature_columns]
        y = df['Price']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        
        return model, encoders, scaler, feature_columns, accuracy

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/car--v1.png", width=80)
    st.title("🚗 CarWale")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Search Cars", "🤖 AI Price Prediction", "📊 Compare Cars", 
         "🧮 EMI Calculator", "⭐ Reviews", "📰 News", "🏪 Dealers"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.info("💡 **Tip:** Use AI Price Prediction for accurate car valuations!")
    
    # Stats
    st.markdown("### 📊 Database Stats")
    st.metric("Brands", len(CAR_DATABASE))
    st.metric("Models", sum(len(info["models"]) for info in CAR_DATABASE.values()))
    
    categories = set(info["category"] for info in CAR_DATABASE.values())
    st.metric("Categories", len(categories))

# Main Content
if page == "🏠 Home":
    # Hero Section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 60px 20px; border-radius: 20px; text-align: center; color: white; margin-bottom: 30px;'>
        <h1 style='font-size: 48px; margin-bottom: 10px; color: white;'>Find Your Dream Car</h1>
        <p style='font-size: 20px; opacity: 0.9;'>Search from 25+ Brands, 100+ Models, New & Used Cars</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='stat-card'>
            <h2 style='color: #667eea; font-size: 36px; margin-bottom: 5px;'>25+</h2>
            <p style='color: #666; margin: 0;'>Car Brands</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='stat-card'>
            <h2 style='color: #e74c3c; font-size: 36px; margin-bottom: 5px;'>100+</h2>
            <p style='color: #666; margin: 0;'>Car Models</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='stat-card'>
            <h2 style='color: #2ecc71; font-size: 36px; margin-bottom: 5px;'>AI</h2>
            <p style='color: #666; margin: 0;'>Powered</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='stat-card'>
            <h2 style='color: #f39c12; font-size: 36px; margin-bottom: 5px;'>₹3.5L-₹12Cr</h2>
            <p style='color: #666; margin: 0;'>Price Range</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Popular Brands
    st.subheader("🔥 Popular Brands")
    brands = list(CAR_DATABASE.keys())[:12]
    cols = st.columns(6)
    for idx, brand in enumerate(brands):
        with cols[idx % 6]:
            st.markdown(f"""
            <div class='car-card' style='text-align: center; padding: 15px;'>
                <div style='font-size: 32px; margin-bottom: 8px;'>🚗</div>
                <strong>{brand}</strong>
                <div style='font-size: 12px; color: #666;'>{CAR_DATABASE[brand]['category']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Featured Cars
    st.subheader("⭐ Featured Cars")
    featured_cars = [
        {"brand": "Maruti Suzuki", "model": "Swift", "price": 599000},
        {"brand": "Hyundai", "model": "Creta", "price": 1099000},
        {"brand": "Tata", "model": "Nexon", "price": 799000},
        {"brand": "Mahindra", "model": "Thar", "price": 1049000},
    ]
    
    cols = st.columns(4)
    for idx, car in enumerate(featured_cars):
        with cols[idx]:
            st.markdown(f"""
            <div class='car-card'>
                <div style='font-size: 48px; text-align: center; margin-bottom: 10px;'>🚗</div>
                <h4 style='margin: 0 0 5px 0;'>{car['model']}</h4>
                <p style='color: #666; font-size: 13px; margin: 0 0 10px 0;'>{car['brand']}</p>
                <h3 style='color: #e74c3c; margin: 0;'>{format_price(car['price'])}</h3>
            </div>
            """, unsafe_allow_html=True)

elif page == "🔍 Search Cars":
    st.title("🔍 Search Cars")
    
    tab1, tab2 = st.tabs(["New Cars", "Used Cars"])
    
    with tab1:
        st.subheader("Search New Cars")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            search_brand = st.selectbox("Select Brand", ["All Brands"] + list(CAR_DATABASE.keys()))
        with col2:
            if search_brand != "All Brands":
                models = CAR_DATABASE[search_brand]["models"]
                search_model = st.selectbox("Select Model", ["All Models"] + models)
            else:
                search_model = st.selectbox("Select Model", ["All Models"])
        with col3:
            budget = st.selectbox("Budget", [
                "All Budgets",
                "Under 5 Lakhs",
                "5-10 Lakhs",
                "10-20 Lakhs",
                "20-50 Lakhs",
                "Above 50 Lakhs"
            ])
        
        if st.button("🔍 Search", key="search_new"):
            st.success("✅ Search Results")
            
            # Display results
            if search_brand != "All Brands":
                brand_info = CAR_DATABASE[search_brand]
                models = brand_info["models"] if search_model == "All Models" else [search_model]
                
                cols = st.columns(3)
                for idx, model in enumerate(models[:9]):
                    with cols[idx % 3]:
                        avg_price = sum(brand_info["price_range"]) / 2
                        st.markdown(f"""
                        <div class='car-card'>
                            <div style='font-size: 48px; text-align: center; margin-bottom: 10px;'>🚗</div>
                            <h4>{search_brand} {model}</h4>
                            <p style='color: #666; font-size: 13px;'>{brand_info['category']}</p>
                            <h3 style='color: #e74c3c;'>{format_price(avg_price)}</h3>
                        </div>
                        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Search Used Cars")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.selectbox("Select City", ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Pune", "Chennai"])
        with col2:
            used_brand = st.selectbox("Brand", ["All Brands"] + list(CAR_DATABASE.keys()), key="used_brand")
        with col3:
            used_budget = st.selectbox("Budget", [
                "All Budgets",
                "Under 2 Lakhs",
                "2-5 Lakhs",
                "5-10 Lakhs",
                "10-20 Lakhs",
                "Above 20 Lakhs"
            ])
        
        if st.button("🔍 Search Used Cars"):
            st.info(f"🔍 Searching for used cars in {city}...")
            st.warning("💡 Use AI Price Prediction to get accurate valuations for used cars!")

elif page == "🤖 AI Price Prediction":
    st.title("🤖 AI-Powered Price Prediction")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
                padding: 20px; border-radius: 12px; margin-bottom: 30px; border-left: 5px solid #667eea;'>
        <h3 style='color: #667eea; margin: 0 0 10px 0;'>🎯 Get Accurate Price Predictions</h3>
        <p style='margin: 0; color: #666;'>Our AI model analyzes 10+ factors to provide accurate car valuations with 85-95% accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Train model if not trained
    if st.session_state.ml_model is None:
        model, encoders, scaler, feature_columns, accuracy = train_ml_model()
        st.session_state.ml_model = model
        st.session_state.encoders = encoders
        st.session_state.scaler = scaler
        st.session_state.feature_columns = feature_columns
        st.success(f"✅ AI Model Trained Successfully! Accuracy: {accuracy*100:.1f}%")
    
    # Prediction Form
    st.subheader("📝 Enter Car Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚗 Car Information")
        brand = st.selectbox("Brand", list(CAR_DATABASE.keys()))
        model_name = st.selectbox("Model", CAR_DATABASE[brand]["models"])
        year = st.selectbox("Year", list(range(datetime.now().year, 2009, -1)))
        mileage = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000, step=1000)
    
    with col2:
        st.markdown("#### ⚙️ Specifications")
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid", "CNG"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic", "CVT", "DCT", "AMT"])
        owners = st.selectbox("Number of Owners", [1, 2, 3, 4])
        condition = st.selectbox("Condition", ["Excellent", "Good", "Fair", "Poor"])
    
    col3, col4 = st.columns(2)
    with col3:
        accident = st.selectbox("Accident History", ["No", "Minor", "Yes"])
    with col4:
        color = st.selectbox("Color", ["White", "Black", "Silver", "Red", "Blue", "Grey", "Brown"])
    
    st.markdown("---")
    
    if st.button("🎯 Predict Price", type="primary", use_container_width=True):
        # Prepare input
        current_year = datetime.now().year
        car_age = current_year - year
        brand_avg_price = sum(CAR_DATABASE[brand]["price_range"]) / 2
        
        input_data = {
            'Year': year,
            'Mileage': mileage,
            'Owners': owners,
            'Car_Age': car_age,
            'Brand_Avg_Price': brand_avg_price
        }
        
        categorical_data = {
            'Brand': brand,
            'Model': model_name,
            'Fuel_Type': fuel,
            'Transmission': transmission,
            'Color': color,
            'Condition': condition,
            'Accident_History': accident,
            'Category': CAR_DATABASE[brand]["category"]
        }
        
        for col, value in categorical_data.items():
            if col in st.session_state.encoders:
                try:
                    encoded_value = st.session_state.encoders[col].transform([value])[0]
                except:
                    encoded_value = 0
                input_data[col + '_Encoded'] = encoded_value
        
        input_df = pd.DataFrame([input_data])
        input_df = input_df[st.session_state.feature_columns]
        input_scaled = st.session_state.scaler.transform(input_df)
        
        predicted_price = st.session_state.ml_model.predict(input_scaled)[0]
        min_price = predicted_price * 0.85
        max_price = predicted_price * 1.15
        
        original_price = sum(CAR_DATABASE[brand]["price_range"]) / 2
        depreciation = ((original_price - predicted_price) / original_price) * 100
        
        # Display Results
        st.success("✅ Prediction Complete!")
        st.markdown("---")
        
        st.markdown(f"### 🚗 {brand} {model_name} ({year})")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='car-card' style='text-align: center;'>
                <h4 style='color: #666; font-size: 14px; margin-bottom: 10px;'>QUICK SALE</h4>
                <h2 style='color: #e74c3c; margin: 0;'>""" + format_price(min_price) + """</h2>
                <p style='color: #999; font-size: 12px; margin-top: 10px;'>Sell within 2 weeks</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='price-card'>
                <h4 style='color: #667eea; font-size: 14px; margin-bottom: 10px;'>FAIR VALUE ⭐</h4>
                <h2 style='color: #667eea; margin: 0;'>""" + format_price(predicted_price) + """</h2>
                <p style='color: #667eea; font-size: 12px; margin-top: 10px;'>Recommended price</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='car-card' style='text-align: center;'>
                <h4 style='color: #666; font-size: 14px; margin-bottom: 10px;'>PREMIUM</h4>
                <h2 style='color: #2ecc71; margin: 0;'>""" + format_price(max_price) + """</h2>
                <p style='color: #999; font-size: 12px; margin-top: 10px;'>Patient sale (1-2 months)</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Details
        st.subheader("📋 Car Details")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Brand:** {brand}")
            st.write(f"**Model:** {model_name}")
            st.write(f"**Year:** {year}")
            st.write(f"**Mileage:** {mileage:,} km")
            st.write(f"**Fuel:** {fuel}")
        with col2:
            st.write(f"**Transmission:** {transmission}")
            st.write(f"**Owners:** {owners}")
            st.write(f"**Condition:** {condition}")
            st.write(f"**Accident:** {accident}")
            st.write(f"**Depreciation:** {depreciation:.1f}%")
        
        # Price Chart
        st.markdown("---")
        st.subheader("📊 Price Breakdown")
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Quick Sale', 'Fair Value', 'Premium'],
                y=[min_price, predicted_price, max_price],
                marker_color=['#e74c3c', '#667eea', '#2ecc71'],
                text=[format_price(min_price), format_price(predicted_price), format_price(max_price)],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="Price Range Analysis",
            xaxis_title="Price Type",
            yaxis_title="Price (₹)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Save prediction
        st.session_state.predictions_history.append({
            'Brand': brand,
            'Model': model_name,
            'Year': year,
            'Predicted Price': format_price(predicted_price),
            'Range': f"{format_price(min_price)} - {format_price(max_price)}",
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        
        st.balloons()

elif page == "📊 Compare Cars":
    st.title("📊 Compare Cars")
    
    num_cars = st.slider("Number of cars to compare", 2, 4, 2)
    
    st.markdown("### Select Cars to Compare")
    
    cols = st.columns(num_cars)
    comparison_data = []
    
    for i in range(num_cars):
        with cols[i]:
            st.markdown(f"#### Car {i+1}")
            brand = st.selectbox(f"Brand", list(CAR_DATABASE.keys()), key=f"comp_brand_{i}")
            model = st.selectbox(f"Model", CAR_DATABASE[brand]["models"], key=f"comp_model_{i}")
            
            avg_price = sum(CAR_DATABASE[brand]["price_range"]) / 2
            comparison_data.append({
                'Car': f"{brand} {model}",
                'Brand': brand,
                'Model': model,
                'Category': CAR_DATABASE[brand]["category"],
                'Avg Price': avg_price
            })
    
    if st.button("Compare Now", type="primary", use_container_width=True):
        st.markdown("---")
        st.subheader("Comparison Results")
        
        # Price Chart
        fig = go.Figure(data=[
            go.Bar(
                x=[d['Car'] for d in comparison_data],
                y=[d['Avg Price'] for d in comparison_data],
                marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe'][:num_cars],
                text=[format_price(d['Avg Price']) for d in comparison_data],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="Price Comparison",
            xaxis_title="Car",
            yaxis_title="Average Price (₹)",
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Details Table
        st.markdown("---")
        st.subheader("Detailed Comparison")
        
        df_compare = pd.DataFrame(comparison_data)
        df_compare['Avg Price'] = df_compare['Avg Price'].apply(format_price)
        st.dataframe(df_compare[['Car', 'Category', 'Avg Price']], use_container_width=True, hide_index=True)
        
        # Best Value
        best_idx = min(range(len(comparison_data)), key=lambda i: comparison_data[i]['Avg Price'])
        st.success(f"💰 **Best Value:** {comparison_data[best_idx]['Car']} at {format_price(comparison_data[best_idx]['Avg Price'])}")

elif page == "🧮 EMI Calculator":
    st.title("🧮 EMI Calculator")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
                padding: 20px; border-radius: 12px; margin-bottom: 30px;'>
        <h3 style='color: #667eea; margin: 0 0 10px 0;'>💰 Calculate Your Car Loan EMI</h3>
        <p style='margin: 0; color: #666;'>Plan your budget with our easy EMI calculator</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Loan Details")
        car_price = st.number_input("Car Price (₹)", min_value=100000, max_value=50000000, value=1000000, step=50000)
        down_payment = st.slider("Down Payment (%)", 0, 50, 20, 5)
        interest_rate = st.slider("Interest Rate (% p.a.)", 5.0, 18.0, 9.5, 0.5)
        tenure = st.slider("Loan Tenure (years)", 1, 7, 5, 1)
        
        if st.button("Calculate EMI", type="primary", use_container_width=True):
            # Calculate
            down_amount = (car_price * down_payment) / 100
            loan_amount = car_price - down_amount
            monthly_rate = interest_rate / (12 * 100)
            months = tenure * 12
            
            emi = (loan_amount * monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
            total_amount = emi * months
            total_interest = total_amount - loan_amount
            
            st.session_state.emi_calculated = True
            st.session_state.emi = emi
            st.session_state.total_amount = total_amount
            st.session_state.total_interest = total_interest
            st.session_state.loan_amount = loan_amount
    
    with col2:
        if 'emi_calculated' in st.session_state and st.session_state.emi_calculated:
            st.subheader("EMI Breakdown")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Monthly EMI", f"₹{st.session_state.emi:,.0f}")
                st.metric("Loan Amount", f"₹{st.session_state.loan_amount:,.0f}")
            with col_b:
                st.metric("Total Amount", f"₹{st.session_state.total_amount:,.0f}")
                st.metric("Total Interest", f"₹{st.session_state.total_interest:,.0f}")
            
            # Pie Chart
            fig = go.Figure(data=[go.Pie(
                labels=['Principal', 'Interest'],
                values=[st.session_state.loan_amount, st.session_state.total_interest],
                hole=.3,
                marker_colors=['#667eea', '#e74c3c']
            )])
            fig.update_layout(title="Loan Breakdown", height=300)
            st.plotly_chart(fig, use_container_width=True)

elif page == "⭐ Reviews":
    st.title("⭐ Expert Reviews")
    
    reviews = [
        {
            "car": "Hyundai Creta",
            "rating": 4.5,
            "reviewer": "Expert Team",
            "text": "The Creta continues to be a strong contender in the compact SUV segment with its premium features and comfortable ride."
        },
        {
            "car": "Mahindra Thar",
            "rating": 4.8,
            "reviewer": "Off-Road Expert",
            "text": "An authentic off-roader that delivers on its promise. Perfect for adventure enthusiasts."
        },
        {
            "car": "Maruti Swift",
            "rating": 4.3,
            "reviewer": "City Reviewer",
            "text": "A practical hatchback with excellent fuel efficiency and low maintenance costs."
        },
        {
            "car": "Tata Nexon",
            "rating": 4.4,
            "reviewer": "Safety Expert",
            "text": "5-star safety rating, feature-loaded, and great value for money. A complete package."
        }
    ]
    
    cols = st.columns(2)
    for idx, review in enumerate(reviews):
        with cols[idx % 2]:
            st.markdown(f"""
            <div class='car-card'>
                <h3>{review['car']}</h3>
                <div style='color: #f39c12; font-size: 18px; margin: 10px 0;'>
                    {'⭐' * int(review['rating'])} {review['rating']}/5
                </div>
                <p style='font-style: italic; color: #666;'>"{review['text']}"</p>
                <p style='color: #999; font-size: 13px; margin-top: 10px;'>- {review['reviewer']}</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "📰 News":
    st.title("📰 Latest Car News")
    
    news = [
        {
            "title": "New Mercedes-Benz E-Class Launched in India",
            "date": "Oct 27, 2024",
            "category": "Launches"
        },
        {
            "title": "Tata Nexon EV Gets Major Price Cut",
            "date": "Oct 26, 2024",
            "category": "Updates"
        },
        {
            "title": "Upcoming Cars in November 2024",
            "date": "Oct 25, 2024",
            "category": "News"
        },
        {
            "title": "Mahindra Thar 5-Door Spotted Testing",
            "date": "Oct 24, 2024",
            "category": "Spy Shots"
        }
    ]
    
    for article in news:
        st.markdown(f"""
        <div class='car-card'>
            <div style='display: flex; gap: 10px; font-size: 12px; color: #999; margin-bottom: 10px;'>
                <span>📅 {article['date']}</span>
                <span>🏷️ {article['category']}</span>
            </div>
            <h3 style='color: #2c3e50; margin: 0 0 10px 0;'>{article['title']}</h3>
            <p style='color: #666; margin: 0;'>Click to read the complete article about latest automotive news and updates.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🏪 Dealers":
    st.title("🏪 Find Car Dealers")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        dealer_city = st.selectbox("Select City", ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Pune", "Chennai"])
    with col2:
        dealer_brand = st.selectbox("Select Brand", ["All Brands"] + list(CAR_DATABASE.keys()))
    with col3:
        st.write("")
        st.write("")
        if st.button("Find Dealers", use_container_width=True):
            st.success(f"🔍 Found 15 authorized dealers for {dealer_brand if dealer_brand != 'All Brands' else 'all brands'} in {dealer_city}")

# Footer
st.markdown("---")
if st.session_state.predictions_history:
    with st.expander("📜 Prediction History"):
        df_history = pd.DataFrame(st.session_state.predictions_history)
        st.dataframe(df_history, use_container_width=True, hide_index=True)

st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <p><strong>🚗 CarWale Clone</strong> | Made with ❤️ using Streamlit</p>
    <p>Complete Car Portal with AI-Powered Price Predictions</p>
</div>
""", unsafe_allow_html=True)
