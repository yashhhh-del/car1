# ======================================================
# SMART CAR PRICING SYSTEM - COMPLETE CAR DATABASE
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
# COMPREHENSIVE CAR DATABASE FOR MANUAL INPUT
# ========================================

CAR_DATABASE = {
    'Maruti Suzuki': {
        'models': ['Alto', 'Alto K10', 'S-Presso', 'Celerio', 'Wagon R', 'Ignis', 'Swift', 'Baleno', 'Dzire', 'Ciaz', 
                  'Ertiga', 'XL6', 'Vitara Brezza', 'Jimny', 'Fronx', 'Grand Vitara', 'Eeco', 'Omni', 'Celerio X'],
        'car_types': ['Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Sedan', 'Sedan',
                     'MUV', 'MUV', 'SUV', 'SUV', 'SUV', 'SUV', 'Van', 'Van', 'Hatchback'],
        'engine_cc': [796, 998, 998, 998, 998, 1197, 1197, 1197, 1197, 1462,
                     1462, 1462, 1462, 1462, 1197, 1462, 1196, 796, 998],
        'power_hp': [48, 67, 67, 67, 67, 83, 90, 90, 90, 103,
                    103, 103, 103, 103, 90, 103, 73, 35, 67],
        'seats': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                 7, 6, 5, 5, 5, 5, 5, 8, 5]
    },
    'Hyundai': {
        'models': ['i10', 'i20', 'Aura', 'Grand i10 Nios', 'Verna', 'Creta', 'Venue', 'Alcazar', 'Tucson', 'Kona Electric',
                  'Santro', 'Xcent', 'Elantra', 'Ioniq 5'],
        'car_types': ['Hatchback', 'Hatchback', 'Sedan', 'Hatchback', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV',
                     'Hatchback', 'Sedan', 'Sedan', 'SUV'],
        'engine_cc': [1086, 1197, 1197, 1197, 1493, 1493, 1197, 2199, 2199, 0,
                     1086, 1197, 1999, 0],
        'power_hp': [69, 83, 83, 83, 115, 115, 83, 148, 148, 136,
                    69, 83, 152, 217],
        'seats': [5, 5, 5, 5, 5, 5, 5, 6, 5, 5,
                 5, 5, 5, 5]
    },
    'Tata': {
        'models': ['Tiago', 'Tigor', 'Altroz', 'Nexon', 'Punch', 'Harrier', 'Safari', 'Nexon EV', 'Tigor EV', 'Tiago EV',
                  'Indica', 'Indigo', 'Sumo', 'Hexa'],
        'car_types': ['Hatchback', 'Sedan', 'Hatchback', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'Sedan', 'Hatchback',
                     'Hatchback', 'Sedan', 'SUV', 'SUV'],
        'engine_cc': [1199, 1199, 1199, 1199, 1199, 1956, 1956, 0, 0, 0,
                     1405, 1405, 2179, 2179],
        'power_hp': [85, 85, 85, 120, 120, 170, 170, 129, 75, 75,
                    70, 70, 120, 156],
        'seats': [5, 5, 5, 5, 5, 5, 7, 5, 5, 5,
                 5, 5, 8, 7]
    },
    'Mahindra': {
        'models': ['Bolero', 'Scorpio', 'XUV300', 'XUV400', 'XUV700', 'Thar', 'Marazzo', 'KUV100', 'TUV300', 'Alturas G4',
                  'Bolero Neo', 'Scorpio N', 'Verito', 'Xylo'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'MUV', 'Hatchback', 'SUV', 'SUV',
                     'SUV', 'SUV', 'Sedan', 'MUV'],
        'engine_cc': [1493, 2179, 1197, 0, 1997, 1997, 1497, 1198, 1493, 2157,
                     1493, 1997, 1461, 2179],
        'power_hp': [75, 140, 110, 150, 200, 150, 123, 83, 100, 178,
                    100, 200, 65, 120],
        'seats': [7, 7, 5, 5, 7, 4, 8, 5, 7, 7,
                 7, 7, 5, 8]
    },
    'Toyota': {
        'models': ['Innova Crysta', 'Fortuner', 'Glanza', 'Urban Cruiser Hyryder', 'Camry', 'Vellfire', 'Hilux', 'Etios', 
                  'Etios Liva', 'Yaris', 'Corolla Altis', 'Innova Hycross'],
        'car_types': ['MUV', 'SUV', 'Hatchback', 'SUV', 'Sedan', 'MUV', 'Pickup', 'Sedan',
                     'Hatchback', 'Sedan', 'Sedan', 'MUV'],
        'engine_cc': [2393, 2694, 1197, 1462, 2487, 2494, 2755, 1496,
                     1496, 1496, 1798, 1987],
        'power_hp': [150, 204, 90, 103, 177, 197, 204, 90,
                    90, 107, 140, 186],
        'seats': [7, 7, 5, 5, 5, 7, 5, 5,
                 5, 5, 5, 7]
    },
    'Honda': {
        'models': ['Amaze', 'City', 'Jazz', 'WR-V', 'Elevate', 'Civic', 'CR-V', 'Brio'],
        'car_types': ['Sedan', 'Sedan', 'Hatchback', 'SUV', 'SUV', 'Sedan', 'SUV', 'Hatchback'],
        'engine_cc': [1199, 1498, 1199, 1199, 1498, 1799, 1997, 1198],
        'power_hp': [90, 121, 90, 90, 121, 141, 158, 88],
        'seats': [5, 5, 5, 5, 5, 5, 5, 5]
    },
    'Kia': {
        'models': ['Seltos', 'Sonet', 'Carens', 'Carnival', 'EV6'],
        'car_types': ['SUV', 'SUV', 'MUV', 'MUV', 'SUV'],
        'engine_cc': [1353, 998, 1482, 2199, 0],
        'power_hp': [140, 120, 115, 200, 229],
        'seats': [5, 5, 6, 7, 5]
    },
    'Volkswagen': {
        'models': ['Polo', 'Vento', 'Taigun', 'Virtus', 'Tiguan', 'T-Roc'],
        'car_types': ['Hatchback', 'Sedan', 'SUV', 'Sedan', 'SUV', 'SUV'],
        'engine_cc': [999, 999, 999, 999, 1984, 1498],
        'power_hp': [110, 110, 115, 115, 190, 150],
        'seats': [5, 5, 5, 5, 5, 5]
    },
    'Skoda': {
        'models': ['Rapid', 'Kushaq', 'Slavia', 'Kodiaq', 'Superb', 'Octavia'],
        'car_types': ['Sedan', 'SUV', 'Sedan', 'SUV', 'Sedan', 'Sedan'],
        'engine_cc': [999, 999, 999, 1984, 1984, 1984],
        'power_hp': [110, 115, 115, 190, 190, 190],
        'seats': [5, 5, 5, 7, 5, 5]
    },
    'Renault': {
        'models': ['Kwid', 'Triber', 'Kiger', 'Duster', 'Captur'],
        'car_types': ['Hatchback', 'MUV', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [999, 999, 999, 1498, 1498],
        'power_hp': [68, 72, 100, 106, 106],
        'seats': [5, 7, 5, 5, 5]
    },
    'Nissan': {
        'models': ['Magnite', 'Kicks', 'Sunny', 'Micra', 'Terrano'],
        'car_types': ['SUV', 'SUV', 'Sedan', 'Hatchback', 'SUV'],
        'engine_cc': [999, 1498, 1498, 1198, 1461],
        'power_hp': [100, 106, 99, 77, 110],
        'seats': [5, 5, 5, 5, 5]
    },
    'MG': {
        'models': ['Hector', 'Astor', 'Gloster', 'ZS EV', 'Comet EV'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV', 'Hatchback'],
        'engine_cc': [1451, 1349, 1996, 0, 0],
        'power_hp': [143, 134, 218, 177, 42],
        'seats': [5, 5, 7, 5, 4]
    },
    'Ford': {
        'models': ['EcoSport', 'Endeavour', 'Figo', 'Aspire', 'Freestyle'],
        'car_types': ['SUV', 'SUV', 'Hatchback', 'Sedan', 'Crossover'],
        'engine_cc': [1498, 1996, 1194, 1194, 1194],
        'power_hp': [123, 170, 96, 96, 96],
        'seats': [5, 7, 5, 5, 5]
    },
    'BMW': {
        'models': ['3 Series', '5 Series', '7 Series', 'X1', 'X3', 'X5', 'X7', 'Z4', 'i4', 'iX'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'Convertible', 'Sedan', 'SUV'],
        'engine_cc': [1998, 1998, 2998, 1499, 1998, 2998, 2998, 1998, 0, 0],
        'power_hp': [255, 248, 335, 140, 248, 335, 400, 197, 340, 523],
        'seats': [5, 5, 5, 5, 5, 5, 7, 2, 5, 5]
    },
    'Mercedes-Benz': {
        'models': ['A-Class', 'C-Class', 'E-Class', 'S-Class', 'GLA', 'GLC', 'GLE', 'GLS', 'EQB', 'EQS'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'Sedan'],
        'engine_cc': [1332, 1497, 1991, 2999, 1332, 1991, 1991, 2999, 0, 0],
        'power_hp': [163, 204, 258, 435, 163, 258, 362, 483, 228, 329],
        'seats': [5, 5, 5, 5, 5, 5, 5, 7, 7, 5]
    },
    'Audi': {
        'models': ['A3', 'A4', 'A6', 'A8', 'Q3', 'Q5', 'Q7', 'Q8', 'e-tron', 'RS5'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'Coupe'],
        'engine_cc': [1395, 1984, 1984, 2995, 1395, 1984, 2995, 2995, 0, 2894],
        'power_hp': [150, 190, 245, 340, 150, 245, 340, 340, 355, 450],
        'seats': [5, 5, 5, 5, 5, 5, 7, 5, 5, 4]
    },
    'Lexus': {
        'models': ['ES', 'LS', 'NX', 'RX', 'UX', 'LC'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'Coupe'],
        'engine_cc': [2487, 3445, 2487, 3456, 1987, 4969],
        'power_hp': [215, 422, 194, 295, 169, 471],
        'seats': [5, 5, 5, 5, 5, 4]
    },
    'Jaguar': {
        'models': ['XE', 'XF', 'XJ', 'F-PACE', 'E-PACE', 'I-PACE'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [1997, 1997, 2993, 1997, 1997, 0],
        'power_hp': [247, 247, 335, 247, 247, 400],
        'seats': [5, 5, 5, 5, 5, 5]
    },
    'Land Rover': {
        'models': ['Range Rover', 'Range Rover Sport', 'Range Rover Velar', 'Range Rover Evoque', 'Discovery', 'Defender'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [2996, 2996, 1997, 1997, 2996, 2996],
        'power_hp': [355, 355, 247, 247, 355, 400],
        'seats': [5, 5, 5, 5, 7, 5]
    },
    'Porsche': {
        'models': ['911', 'Panamera', 'Cayenne', 'Macan', 'Taycan'],
        'car_types': ['Coupe', 'Sedan', 'SUV', 'SUV', 'Sedan'],
        'engine_cc': [2981, 2894, 2995, 1984, 0],
        'power_hp': [385, 330, 340, 265, 402],
        'seats': [4, 5, 5, 5, 4]
    },
    'Volvo': {
        'models': ['S60', 'S90', 'XC40', 'XC60', 'XC90', 'C40'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [1969, 1969, 1969, 1969, 1969, 0],
        'power_hp': [250, 250, 197, 250, 300, 231],
        'seats': [5, 5, 5, 5, 7, 5]
    },
    'Jeep': {
        'models': ['Compass', 'Wrangler', 'Grand Cherokee', 'Meridian'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [1956, 1995, 2998, 1956],
        'power_hp': [173, 268, 360, 173],
        'seats': [5, 4, 5, 7]
    },
    'Citroen': {
        'models': ['C3', 'C5 Aircross', 'eC3'],
        'car_types': ['Hatchback', 'SUV', 'Hatchback'],
        'engine_cc': [1199, 1499, 0],
        'power_hp': [82, 130, 57],
        'seats': [5, 5, 5]
    },
    'Mitsubishi': {
        'models': ['Outlander', 'Pajero Sport', 'Eclipse Cross'],
        'car_types': ['SUV', 'SUV', 'SUV'],
        'engine_cc': [2360, 2442, 1469],
        'power_hp': [166, 178, 163],
        'seats': [7, 7, 5]
    },
    'Force': {
        'models': ['Gurkha', 'Motors', 'Traveller'],
        'car_types': ['SUV', 'SUV', 'Van'],
        'engine_cc': [2596, 2596, 2596],
        'power_hp': [80, 80, 80],
        'seats': [5, 5, 13]
    },
    'Isuzu': {
        'models': ['D-Max', 'MU-X', 'V-Cross'],
        'car_types': ['Pickup', 'SUV', 'Pickup'],
        'engine_cc': [1898, 1898, 1898],
        'power_hp': [164, 164, 164],
        'seats': [5, 7, 5]
    },
    'Mini': {
        'models': ['Cooper', 'Countryman', 'Clubman'],
        'car_types': ['Hatchback', 'SUV', 'Hatchback'],
        'engine_cc': [1499, 1499, 1499],
        'power_hp': [136, 136, 136],
        'seats': [4, 5, 4]
    },
    'BYD': {
        'models': ['Atto 3', 'E6', 'Han'],
        'car_types': ['SUV', 'MPV', 'Sedan'],
        'engine_cc': [0, 0, 0],
        'power_hp': [201, 95, 215],
        'seats': [5, 5, 5]
    }
}

FUEL_TYPES = ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
TRANSMISSIONS = ["Manual", "Automatic", "CVT", "DCT", "AMT"]
CAR_CONDITIONS = ["Excellent", "Very Good", "Good", "Fair", "Poor"]
OWNER_TYPES = ["First", "Second", "Third", "Fourth & Above"]
INSURANCE_STATUS = ["Comprehensive", "Third Party", "Expired", "No Insurance"]
COLORS = ["White", "Black", "Silver", "Grey", "Red", "Blue", "Brown", "Green", "Yellow", "Orange", "Purple", "Other"]
CITIES = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune", "Hyderabad", "Kolkata", "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Chandigarh"]

# ========================================
# ENHANCED MANUAL INPUT FORM
# ========================================

def show_manual_input_form():
    """Show comprehensive manual input form for car details"""
    st.subheader("🔧 Complete Car Details Entry")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand selection with search
        brand = st.selectbox("Brand", list(CAR_DATABASE.keys()), 
                           help="Select car brand from comprehensive database")
        
        if brand in CAR_DATABASE:
            model = st.selectbox("Model", CAR_DATABASE[brand]['models'],
                               help=f"Select {brand} model")
            
            # Auto-fill technical specifications
            if model in CAR_DATABASE[brand]['models']:
                model_index = CAR_DATABASE[brand]['models'].index(model)
                car_type = CAR_DATABASE[brand]['car_types'][model_index]
                engine_cc = CAR_DATABASE[brand]['engine_cc'][model_index]
                power_hp = CAR_DATABASE[brand]['power_hp'][model_index]
                seats = CAR_DATABASE[brand]['seats'][model_index]
                
                st.text_input("Car Type", value=car_type, disabled=True)
                st.text_input("Engine Capacity", value=f"{engine_cc} cc", disabled=True)
                st.text_input("Power", value=f"{power_hp} HP", disabled=True)
                st.text_input("Seating Capacity", value=f"{seats} seats", disabled=True)
            else:
                car_type = st.selectbox("Car Type", ["Hatchback", "Sedan", "SUV", "MUV", "Coupe", "Convertible", "Van", "Pickup"])
                engine_cc = st.number_input("Engine CC", min_value=600, max_value=5000, value=1200)
                power_hp = st.number_input("Power (HP)", min_value=40, max_value=500, value=80)
                seats = st.number_input("Seats", min_value=2, max_value=9, value=5)
        else:
            model = st.text_input("Model Name", placeholder="Enter model name")
            car_type = st.selectbox("Car Type", ["Hatchback", "Sedan", "SUV", "MUV", "Coupe", "Convertible", "Van", "Pickup"])
            engine_cc = st.number_input("Engine CC", min_value=600, max_value=5000, value=1200)
            power_hp = st.number_input("Power (HP)", min_value=40, max_value=500, value=80)
            seats = st.number_input("Seats", min_value=2, max_value=9, value=5)
        
        current_year = datetime.now().year
        year = st.number_input("Manufacturing Year", min_value=1990, max_value=current_year, 
                             value=current_year-3, help="Year when car was manufactured")
        
        fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
        transmission = st.selectbox("Transmission", TRANSMISSIONS)
    
    with col2:
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=30000, step=1000,
                                help="Total kilometers driven")
        
        color = st.selectbox("Color", COLORS)
        condition = st.selectbox("Car Condition", CAR_CONDITIONS,
                               help="Overall condition of the vehicle")
        
        owner_type = st.selectbox("Owner Type", OWNER_TYPES,
                                help="Number of previous owners")
        insurance_status = st.selectbox("Insurance Status", INSURANCE_STATUS)
        
        registration_city = st.selectbox("Registration City", CITIES,
                                       help="City where car is registered")
    
    # Additional details section
    st.subheader("📋 Additional Details")
    
    col3, col4 = st.columns(2)
    
    with col3:
        service_history = st.radio("Service History", 
                                 ["Full Service History", "Partial Service History", "No Service History"])
        
        accident_history = st.radio("Accident History", 
                                  ["No Accidents", "Minor Accidents", "Major Accidents"])
    
    with col4:
        car_availability = st.radio("Car Availability", ["Available", "Sold"])
        
        # Additional features
        features = st.multiselect("Additional Features",
                                ["Power Steering", "Power Windows", "Air Conditioning", "Music System",
                                 "Alloy Wheels", "Sunroof", "Leather Seats", "Rear Camera", "GPS Navigation",
                                 "Keyless Entry", "Push Start", "ABS", "Airbags", "ESP"])
    
    # Generate unique Car_ID
    car_id = f"{brand[:3].upper()}_{model[:3].upper()}_{year}_{np.random.randint(1000,9999)}"
    
    # Return all input data
    input_data = {
        'Car_ID': car_id,
        'Brand': brand,
        'Model': model,
        'Car_Type': car_type,
        'Year': year,
        'Fuel_Type': fuel_type,
        'Transmission': transmission,
        'Mileage': mileage,
        'Engine_cc': engine_cc,
        'Power_HP': power_hp,
        'Seats': seats,
        'Color': color,
        'Condition': condition,
        'Owner_Type': owner_type,
        'Insurance_Status': insurance_status,
        'Registration_City': registration_city,
        'Service_History': service_history,
        'Accident_History': accident_history,
        'Car_Availability': car_availability,
        'Features': ', '.join(features) if features else 'None'
    }
    
    # Show summary
    with st.expander("📊 Car Details Summary", expanded=True):
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write(f"**Brand:** {brand}")
            st.write(f"**Model:** {model}")
            st.write(f"**Year:** {year}")
            st.write(f"**Fuel Type:** {fuel_type}")
            st.write(f"**Transmission:** {transmission}")
            
        with summary_col2:
            st.write(f"**Mileage:** {mileage:,} km")
            st.write(f"**Engine:** {engine_cc} cc")
            st.write(f"**Power:** {power_hp} HP")
            st.write(f"**Condition:** {condition}")
            st.write(f"**Owners:** {owner_type}")
    
    return input_data

# ========================================
# BRAND STATISTICS FUNCTION
# ========================================

def show_brand_statistics():
    """Show statistics about available car brands"""
    st.sidebar.subheader("📈 Brand Statistics")
    
    total_brands = len(CAR_DATABASE)
    total_models = sum(len(CAR_DATABASE[brand]['models']) for brand in CAR_DATABASE)
    
    st.sidebar.info(f"""
    **Database Overview:**
    - 🚗 **Brands:** {total_brands}
    - 🎯 **Models:** {total_models}
    - 📊 **Coverage:** Comprehensive
    """)
    
    # Brand distribution
    brand_counts = {brand: len(CAR_DATABASE[brand]['models']) for brand in CAR_DATABASE}
    top_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    st.sidebar.write("**Top 5 Brands by Models:**")
    for brand, count in top_brands:
        st.sidebar.write(f"- {brand}: {count} models")

# ========================================
# CAR SEARCH FUNCTIONALITY
# ========================================

def search_cars():
    """Search functionality for cars in database"""
    st.sidebar.subheader("🔍 Search Cars")
    
    search_brand = st.sidebar.selectbox("Search by Brand", ["All"] + list(CAR_DATABASE.keys()))
    search_type = st.sidebar.selectbox("Search by Type", ["All", "Hatchback", "Sedan", "SUV", "MUV"])
    
    if st.sidebar.button("Search"):
        results = []
        
        for brand in CAR_DATABASE:
            if search_brand != "All" and brand != search_brand:
                continue
                
            for i, model in enumerate(CAR_DATABASE[brand]['models']):
                car_type = CAR_DATABASE[brand]['car_types'][i]
                
                if search_type != "All" and car_type != search_type:
                    continue
                    
                results.append({
                    'Brand': brand,
                    'Model': model,
                    'Type': car_type,
                    'Engine': CAR_DATABASE[brand]['engine_cc'][i],
                    'Power': CAR_DATABASE[brand]['power_hp'][i],
                    'Seats': CAR_DATABASE[brand]['seats'][i]
                })
        
        if results:
            st.sidebar.success(f"Found {len(results)} cars")
            df_results = pd.DataFrame(results)
            st.sidebar.dataframe(df_results, use_container_width=True)
        else:
            st.sidebar.warning("No cars found matching criteria")

# ========================================
# ENHANCED LIVE PRICE DATABASE
# ========================================

@st.cache_data(ttl=3600)
def get_enhanced_live_prices(brand, model):
    """Get enhanced live prices for all car models"""
    
    # Comprehensive price database
    car_price_database = {
        'Maruti Suzuki': {
            'Alto': [150000, 250000, 350000],
            'Swift': [300000, 450000, 600000],
            'Baleno': [350000, 500000, 700000],
            'Dzire': [320000, 480000, 650000],
            'Vitara Brezza': [500000, 700000, 900000],
            'Ertiga': [450000, 650000, 850000],
            'Wagon R': [200000, 300000, 400000],
            'Celerio': [250000, 350000, 450000],
            'Ciaz': [450000, 650000, 850000],
            'S-Presso': [280000, 380000, 480000],
            'Ignis': [320000, 450000, 580000],
            'XL6': [550000, 750000, 950000],
            'Grand Vitara': [800000, 1100000, 1400000],
            'Fronx': [450000, 600000, 800000],
            'Jimny': [600000, 800000, 1000000]
        },
        'Hyundai': {
            'i10': [250000, 350000, 450000],
            'i20': [350000, 500000, 650000],
            'Creta': [600000, 850000, 1100000],
            'Verna': [450000, 650000, 850000],
            'Venue': [450000, 600000, 800000],
            'Aura': [320000, 450000, 580000],
            'Alcazar': [800000, 1100000, 1400000],
            'Tucson': [1200000, 1600000, 2000000],
            'Grand i10 Nios': [300000, 420000, 550000]
        },
        'Tata': {
            'Tiago': [250000, 350000, 450000],
            'Nexon': [450000, 650000, 850000],
            'Altroz': [350000, 500000, 650000],
            'Harrier': [800000, 1100000, 1400000],
            'Safari': [900000, 1200000, 1500000],
            'Punch': [300000, 450000, 600000],
            'Tigor': [280000, 400000, 520000]
        },
        'Mahindra': {
            'Scorpio': [500000, 700000, 900000],
            'XUV300': [450000, 600000, 800000],
            'XUV700': [900000, 1200000, 1500000],
            'Thar': [600000, 850000, 1100000],
            'Bolero': [300000, 450000, 600000],
            'Marazzo': [500000, 700000, 900000]
        },
        'Toyota': {
            'Innova Crysta': [1000000, 1400000, 1800000],
            'Fortuner': [1500000, 2000000, 2500000],
            'Glanza': [350000, 500000, 650000],
            'Urban Cruiser Hyryder': [600000, 800000, 1000000],
            'Camry': [1800000, 2300000, 2800000]
        },
        'Honda': {
            'City': [450000, 650000, 850000],
            'Amaze': [350000, 500000, 650000],
            'WR-V': [400000, 550000, 700000],
            'Elevate': [600000, 800000, 1000000]
        },
        'Kia': {
            'Seltos': [600000, 800000, 1000000],
            'Sonet': [450000, 600000, 800000],
            'Carens': [650000, 850000, 1100000]
        },
        'Volkswagen': {
            'Polo': [350000, 500000, 650000],
            'Vento': [400000, 550000, 700000],
            'Taigun': [600000, 800000, 1000000],
            'Virtus': [550000, 750000, 950000]
        }
    }
    
    # Default price ranges for brands not in database
    brand_defaults = {
        'BMW': [1500000, 2500000, 4000000],
        'Mercedes-Benz': [1800000, 3000000, 5000000],
        'Audi': [1600000, 2800000, 4500000],
        'Lexus': [2000000, 3500000, 5500000],
        'Jaguar': [2200000, 3800000, 6000000],
        'Land Rover': [2500000, 4500000, 7000000],
        'Porsche': [5000000, 8000000, 12000000],
        'Volvo': [1200000, 2000000, 3500000],
        'Jeep': [800000, 1200000, 1800000],
        'MG': [700000, 1000000, 1400000]
    }
    
    try:
        if brand in car_price_database and model in car_price_database[brand]:
            prices = car_price_database[brand][model]
            sources = ["Used Car Market Database"]
        elif brand in brand_defaults:
            prices = brand_defaults[brand]
            sources = ["Luxury Car Market Average"]
        else:
            # Estimate based on car type
            base_prices = {
                'Hatchback': [200000, 350000, 500000],
                'Sedan': [300000, 500000, 700000],
                'SUV': [400000, 650000, 900000],
                'MUV': [350000, 550000, 750000]
            }
            prices = base_prices.get('Sedan', [300000, 500000, 800000])
            sources = ["Market Estimate"]
            
    except Exception as e:
        prices = [300000, 500000, 800000]
        sources = ["General Market Average"]
    
    return prices, sources

# ========================================
# MAIN APPLICATION WITH ENHANCED DATABASE
# ========================================

def main():
    # Initialize session state
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    st.set_page_config(
        page_title="Car Price Predictor - Complete Database", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    st.title("🚗 Car Price Prediction System")
    st.markdown("### **Complete Car Database with All Brands & Models**")
    
    # Show brand statistics in sidebar
    show_brand_statistics()
    
    # Add search functionality
    search_cars()
    
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/48/car.png")
        st.title("Navigation")
        page = st.radio("Go to", ["Price Prediction", "Brand Explorer", "Market Analysis"])
        
        st.markdown("---")
        st.subheader("Database Info")
        st.info(f"**{len(CAR_DATABASE)} brands** • **{sum(len(CAR_DATABASE[brand]['models']) for brand in CAR_DATABASE)} models**")
    
    if page == "Price Prediction":
        st.subheader("💰 Car Price Prediction")
        
        # Manual input form
        input_data = show_manual_input_form()
        
        if input_data:
            brand = input_data['Brand']
            model = input_data['Model']
            
            # Show live prices
            if brand and model:
                with st.spinner(f'🔍 Fetching market prices for {brand} {model}...'):
                    prices, sources = get_enhanced_live_prices(brand, model)
                
                if prices and len(prices) >= 3:
                    min_price, avg_price, max_price = prices
                    
                    st.subheader("💰 Current Market Price Range")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Budget Range", f"₹{min_price:,.0f}", "Basic Condition")
                    
                    with col2:
                        st.metric("Fair Price", f"₹{avg_price:,.0f}", "Good Condition")
                    
                    with col3:
                        st.metric("Premium Range", f"₹{max_price:,.0f}", "Excellent Condition")
                    
                    # Price visualization
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=[min_price, avg_price, max_price],
                        y=['Budget', 'Fair', 'Premium'],
                        orientation='h',
                        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                        text=[f'₹{min_price:,.0f}', f'₹{avg_price:,.0f}', f'₹{max_price:,.0f}'],
                        textposition='auto',
                    ))
                    
                    fig.update_layout(
                        title=f"{brand} {model} - Price Range Analysis",
                        xaxis_title="Price (₹)",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction button
                    if st.button("🎯 Predict Exact Price", type="primary", use_container_width=True):
                        # Calculate final price based on condition and other factors
                        condition_multiplier = {
                            "Excellent": 1.2,
                            "Very Good": 1.1,
                            "Good": 1.0,
                            "Fair": 0.8,
                            "Poor": 0.6
                        }
                        
                        mileage_factor = max(0.5, 1 - (input_data['Mileage'] / 200000))
                        owner_multiplier = {
                            "First": 1.1,
                            "Second": 1.0,
                            "Third": 0.9,
                            "Fourth & Above": 0.8
                        }
                        
                        final_price = avg_price * condition_multiplier[input_data['Condition']] * mileage_factor * owner_multiplier[input_data['Owner_Type']]
                        
                        st.success(f"💎 **Predicted Price: ₹{final_price:,.0f}**")
                        
                        # Price breakdown
                        st.subheader("📊 Price Breakdown")
                        
                        breakdown_data = {
                            'Factor': ['Base Price', 'Condition', 'Mileage', 'Owner History', 'Final Price'],
                            'Multiplier': ['-', f"{condition_multiplier[input_data['Condition']]:.1f}x", 
                                         f"{mileage_factor:.2f}x", f"{owner_multiplier[input_data['Owner_Type']]:.1f}x", '-'],
                            'Amount': [f"₹{avg_price:,.0f}", '-', '-', '-', f"₹{final_price:,.0f}"]
                        }
                        
                        st.table(pd.DataFrame(breakdown_data))
                        
                        st.balloons()
    
    elif page == "Brand Explorer":
        st.subheader("🔍 Car Brand & Model Explorer")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_brand = st.selectbox("Select Brand", list(CAR_DATABASE.keys()))
            
            if selected_brand in CAR_DATABASE:
                st.info(f"**{selected_brand}** has **{len(CAR_DATABASE[selected_brand]['models'])}** models")
                
                # Brand statistics
                types_count = {}
                for car_type in CAR_DATABASE[selected_brand]['car_types']:
                    types_count[car_type] = types_count.get(car_type, 0) + 1
                
                st.write("**Model Distribution:**")
                for car_type, count in types_count.items():
                    st.write(f"- {car_type}: {count} models")
        
        with col2:
            if selected_brand in CAR_DATABASE:
                models_data = []
                for i, model in enumerate(CAR_DATABASE[selected_brand]['models']):
                    models_data.append({
                        'Model': model,
                        'Type': CAR_DATABASE[selected_brand]['car_types'][i],
                        'Engine (cc)': CAR_DATABASE[selected_brand]['engine_cc'][i],
                        'Power (HP)': CAR_DATABASE[selected_brand]['power_hp'][i],
                        'Seats': CAR_DATABASE[selected_brand]['seats'][i]
                    })
                
                df_models = pd.DataFrame(models_data)
                st.dataframe(df_models, use_container_width=True)
    
    elif page == "Market Analysis":
        st.subheader("📈 Car Market Analysis")
        
        # Brand distribution
        brand_counts = {brand: len(CAR_DATABASE[brand]['models']) for brand in CAR_DATABASE}
        
        fig1 = px.bar(x=list(brand_counts.keys()), y=list(brand_counts.values()),
                     title="Number of Models per Brand",
                     labels={'x': 'Brand', 'y': 'Number of Models'})
        st.plotly_chart(fig1, use_container_width=True)
        
        # Car type distribution
        type_counts = {}
        for brand in CAR_DATABASE:
            for car_type in CAR_DATABASE[brand]['car_types']:
                type_counts[car_type] = type_counts.get(car_type, 0) + 1
        
        fig2 = px.pie(values=list(type_counts.values()), names=list(type_counts.keys()),
                     title="Car Type Distribution")
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
