# ======================================================
# SMART PRICING SYSTEM FOR USED CARS - SIMPLE VERSION
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# Page config
st.set_page_config(page_title="Smart Car Pricing PRO", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚗 Smart Car Pricing System PRO</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Price Predictions, EMI Calculator, Comparison Tool & More!")

sns.set(style="whitegrid")

# Sidebar navigation
with st.sidebar:
    st.title("📊 Navigation")
    page = st.radio("Go to", ["🏠 Home", "💰 Price Prediction", "📊 Compare Cars", "🧮 EMI Calculator", "📈 Market Insights", "📥 Download Report"])
    
    st.markdown("---")
    st.info("💡 Upload your dataset to unlock all features!")

# File Upload
uploaded_file = st.file_uploader("📂 Upload CSV/XLSX File", type=["csv","xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            import openpyxl
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    if 'Market_Price(INR)' not in df.columns:
        st.error("❌ Dataset must include 'Market_Price(INR)' column.")
        st.stop()

    # Data Preprocessing
    df_clean = df.dropna()
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    encoders = {}
    df_encoded = df_clean.copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_clean[col].astype(str))
        encoders[col] = le

    # Model Training
    X = df_encoded.drop(columns=['Market_Price(INR)'])
    y = df_encoded['Market_Price(INR)']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2 Score': r2_score(y_test, y_pred)
        }

    result_df = pd.DataFrame(results).T
    best_model_name = result_df['R2 Score'].idxmax()
    best_model = trained_models[best_model_name]

    # ============================================
    # HOME PAGE
    # ============================================
    if page == "🏠 Home":
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cars", f"{len(df_clean):,}")
        with col2:
            st.metric("Unique Brands", f"{df_clean['Brand'].nunique()}")
        with col3:
            st.metric("Avg Price", f"₹{df_clean['Market_Price(INR)'].mean():,.0f}")
        with col4:
            st.metric("Price Range", f"₹{df_clean['Market_Price(INR)'].min():,.0f} - ₹{df_clean['Market_Price(INR)'].max():,.0f}")

        st.markdown("---")
        
        # Top Stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 10 Most Popular Brands")
            brand_counts = df_clean['Brand'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=brand_counts.values, y=brand_counts.index, palette='viridis', ax=ax)
            ax.set_xlabel('Number of Cars')
            ax.set_ylabel('Brand')
            st.pyplot(fig)
        
        with col2:
            st.subheader("💎 Top 10 Most Expensive Cars")
            top_expensive = df_clean.nlargest(10, 'Market_Price(INR)')[['Brand', 'Model', 'Market_Price(INR)']]
            top_expensive['Price'] = top_expensive['Market_Price(INR)'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(top_expensive[['Brand', 'Model', 'Price']], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🔍 Quick Search Dataset")
        
        col1, col2 = st.columns(2)
        with col1:
            search_brand = st.multiselect("Filter by Brand", options=["All"] + sorted(df_clean['Brand'].unique().tolist()))
        with col2:
            if 'Fuel_Type' in df_clean.columns:
                search_fuel = st.multiselect("Filter by Fuel Type", options=["All"] + sorted(df_clean['Fuel_Type'].unique().tolist()))
            else:
                search_fuel = ["All"]
        
        filtered_data = df_clean.copy()
        if search_brand and "All" not in search_brand:
            filtered_data = filtered_data[filtered_data['Brand'].isin(search_brand)]
        if search_fuel and "All" not in search_fuel and 'Fuel_Type' in df_clean.columns:
            filtered_data = filtered_data[filtered_data['Fuel_Type'].isin(search_fuel)]
        
        st.dataframe(filtered_data, use_container_width=True)
        
        # Download filtered data
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Data", csv, "filtered_cars.csv", "text/csv")

    # ============================================
    # PRICE PREDICTION PAGE
    # ============================================
    elif page == "💰 Price Prediction":
        st.subheader("💰 Predict Car Price")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🤖 Model Performance")
            st.dataframe(result_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
            st.success(f"🏆 Best Model: **{best_model_name}**")
        
        with col2:
            st.markdown("### 📊 Model Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            models_list = list(results.keys())
            r2_scores = [results[m]['R2 Score'] for m in models_list]
            sns.barplot(x=r2_scores, y=models_list, palette='coolwarm', ax=ax)
            ax.set_xlabel('R2 Score')
            ax.set_title('Model Performance')
            st.pyplot(fig)

        st.markdown("---")
        
        # Brand and Model Selection
        brands = sorted(df_clean['Brand'].unique())
        selected_brand = st.selectbox("🚘 Select Brand", brands)

        filtered_models = sorted(df_clean[df_clean['Brand'] == selected_brand]['Model'].unique())
        selected_model = st.selectbox("🔧 Select Model", filtered_models)

        filtered_rows = df_clean[(df_clean['Brand'] == selected_brand) & 
                                (df_clean['Model'] == selected_model)]

        # Car Images Gallery
        if len(filtered_rows) > 0:
            st.markdown("### 🖼️ Car Images")
            st.write(f"Showing images for: **{selected_brand} {selected_model}** ({len(filtered_rows)} cars found)")
            
            for i in range(0, min(len(filtered_rows), 6), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(filtered_rows):
                        brand = filtered_rows.iloc[idx]['Brand']
                        model = filtered_rows.iloc[idx]['Model']
                        year = filtered_rows.iloc[idx]['Year']
                        
                        search_query = f"{brand}+{model}+{year}+car".replace(' ', '+')
                        img_url = f"https://tse1.mm.bing.net/th?q={search_query}&w=600&h=400&c=7&rs=1&p=0&dpr=1&pid=1.7&mkt=en-IN&adlt=moderate"
                        
                        try:
                            col.image(img_url, use_container_width=True, caption=f"{brand} {model} ({year})")
                        except:
                            col.info("Image not available")

            st.markdown("---")
            
            # Auto-fill inputs
            filtered_row = filtered_rows.iloc[0]
            
            st.markdown("### 🧩 Car Details (Editable)")
            
            col1, col2, col3 = st.columns(3)
            inputs = {}
            
            feature_idx = 0
            for col in feature_columns:
                if col in filtered_row.index:
                    with [col1, col2, col3][feature_idx % 3]:
                        if df_clean[col].dtype == 'object':
                            options = sorted(df_clean[col].unique())
                            default = filtered_row[col]
                            inputs[col] = st.selectbox(f"{col}", options, index=options.index(default), key=f"pred_{col}")
                        else:
                            min_val = int(df_clean[col].min())
                            max_val = int(df_clean[col].max())
                            default_val = int(filtered_row[col])
                            inputs[col] = st.slider(f"{col}", min_val, max_val, default_val, key=f"pred_{col}")
                    feature_idx += 1

            # Prediction
            if st.button("🔍 Predict Price", type="primary"):
                input_df = pd.DataFrame([inputs])
                for col in encoders:
                    if col in input_df:
                        input_df[col] = encoders[col].transform(input_df[col].astype(str))
                input_scaled = scaler.transform(input_df)
                predicted_price = best_model.predict(input_scaled)[0]

                st.markdown("---")
                st.subheader("📊 Price Estimation")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Minimum Price", f"₹{predicted_price*0.9:,.0f}", delta="-10%")
                with col2:
                    st.metric("Fair Market Price", f"₹{predicted_price:,.0f}", delta="Recommended")
                with col3:
                    st.metric("Maximum Price", f"₹{predicted_price*1.1:,.0f}", delta="+10%")
                
                st.balloons()
                
                # Save prediction
                if 'predictions' not in st.session_state:
                    st.session_state.predictions = []
                
                st.session_state.predictions.append({
                    'Brand': selected_brand,
                    'Model': selected_model,
                    'Predicted_Price': predicted_price,
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

    # ============================================
    # COMPARE CARS PAGE
    # ============================================
    elif page == "📊 Compare Cars":
        st.subheader("📊 Compare Multiple Cars")
        
        st.info("Select 2-3 cars to compare side-by-side")
        
        num_cars = st.slider("Number of cars to compare", 2, 3, 2)
        
        comparison_data = []
        cols = st.columns(num_cars)
        
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"### Car {i+1}")
                brands = sorted(df_clean['Brand'].unique())
                brand = st.selectbox(f"Brand", brands, key=f"comp_brand_{i}")
                
                models = sorted(df_clean[df_clean['Brand'] == brand]['Model'].unique())
                model = st.selectbox(f"Model", models, key=f"comp_model_{i}")
                
                car_data = df_clean[(df_clean['Brand'] == brand) & (df_clean['Model'] == model)].iloc[0]
                
                # Show car image
                search_query = f"{brand}+{model}+car".replace(' ', '+')
                img_url = f"https://tse1.mm.bing.net/th?q={search_query}&w=400&h=300&c=7&rs=1&p=0&dpr=1&pid=1.7&mkt=en-IN&adlt=moderate"
                try:
                    st.image(img_url, use_container_width=True)
                except:
                    st.info("Image not available")
                
                comparison_data.append({
                    'Brand': brand,
                    'Model': model,
                    'Price': car_data['Market_Price(INR)'],
                    'Year': car_data['Year'],
                    'Fuel_Type': car_data.get('Fuel_Type', 'N/A'),
                    'Transmission': car_data.get('Transmission', 'N/A'),
                    'Mileage': car_data.get('Mileage(km)', 'N/A'),
                    'Power_HP': car_data.get('Power_HP', 'N/A')
                })
        
        if st.button("🔄 Compare Now"):
            st.markdown("---")
            st.subheader("📋 Comparison Results")
            
            comparison_df = pd.DataFrame(comparison_data).T
            comparison_df.columns = [f"Car {i+1}" for i in range(num_cars)]
            st.dataframe(comparison_df, use_container_width=True)
            
            # Price comparison chart
            st.markdown("### 💰 Price Comparison")
            fig, ax = plt.subplots(figsize=(10, 5))
            car_names = [f"{d['Brand']} {d['Model']}" for d in comparison_data]
            prices = [d['Price'] for d in comparison_data]
            sns.barplot(x=car_names, y=prices, palette='Set2', ax=ax)
            ax.set_ylabel('Price (INR)')
            ax.set_title('Price Comparison')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            # Best value
            best_idx = prices.index(min(prices))
            st.success(f"💰 Best Value: **{comparison_data[best_idx]['Brand']} {comparison_data[best_idx]['Model']}** at ₹{comparison_data[best_idx]['Price']:,.0f}")

    # ============================================
    # EMI CALCULATOR PAGE
    # ============================================
    elif page == "🧮 EMI Calculator":
        st.subheader("🧮 EMI Calculator")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Enter Loan Details")
            
            car_price = st.number_input("Car Price (₹)", min_value=100000, max_value=50000000, value=1000000, step=50000)
            down_payment = st.slider("Down Payment (%)", 0, 50, 20)
            interest_rate = st.slider("Annual Interest Rate (%)", 5.0, 20.0, 9.5, step=0.5)
            tenure_years = st.slider("Loan Tenure (Years)", 1, 7, 5)
            
            # Calculate EMI
            principal = car_price - (car_price * down_payment / 100)
            rate_monthly = interest_rate / (12 * 100)
            tenure_months = tenure_years * 12
            
            if rate_monthly > 0:
                emi = principal * rate_monthly * ((1 + rate_monthly)**tenure_months) / (((1 + rate_monthly)**tenure_months) - 1)
            else:
                emi = principal / tenure_months
            
            total_amount = emi * tenure_months
            total_interest = total_amount - principal
        
        with col2:
            st.markdown("### EMI Breakdown")
            
            st.metric("Monthly EMI", f"₹{emi:,.0f}")
            st.metric("Total Amount Payable", f"₹{total_amount:,.0f}")
            st.metric("Total Interest", f"₹{total_interest:,.0f}")
            st.metric("Down Payment", f"₹{car_price * down_payment / 100:,.0f}")
            
            # Pie chart
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie([principal, total_interest], labels=['Principal', 'Interest'], 
                   autopct='%1.1f%%', startangle=90, colors=['#4A90E2', '#E24A4A'])
            ax.set_title('Loan Breakdown')
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Payment schedule
        st.subheader("📅 Payment Schedule (First 12 Months)")
        
        schedule = []
        balance = principal
        
        for month in range(1, min(13, tenure_months + 1)):
            interest_payment = balance * rate_monthly
            principal_payment = emi - interest_payment
            balance -= principal_payment
            
            schedule.append({
                'Month': month,
                'EMI': f"₹{emi:,.0f}",
                'Principal': f"₹{principal_payment:,.0f}",
                'Interest': f"₹{interest_payment:,.0f}",
                'Balance': f"₹{balance:,.0f}"
            })
        
        st.dataframe(pd.DataFrame(schedule), use_container_width=True, hide_index=True)

    # ============================================
    # MARKET INSIGHTS PAGE
    # ============================================
    elif page == "📈 Market Insights":
        st.subheader("📈 Market Insights & Analytics")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Distribution", "⛽ Fuel Analysis", "🏙️ City-wise", "📅 Year Trends"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(df_clean['Market_Price(INR)'], kde=True, bins=50, ax=ax, color='skyblue')
                ax.set_title('Price Distribution')
                ax.set_xlabel('Price (INR)')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(y=df_clean['Market_Price(INR)'], ax=ax, color='lightgreen')
                ax.set_title('Price Range Analysis')
                st.pyplot(fig)
        
        with tab2:
            if 'Fuel_Type' in df_clean.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(data=df_clean, x='Fuel_Type', y='Market_Price(INR)', ax=ax, palette='Set3')
                    ax.set_title('Price by Fuel Type')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                with col2:
                    fuel_counts = df_clean['Fuel_Type'].value_counts()
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(fuel_counts.values, labels=fuel_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.set_title('Fuel Type Distribution')
                    st.pyplot(fig)
        
        with tab3:
            if 'Registration_City' in df_clean.columns:
                city_avg = df_clean.groupby('Registration_City')['Market_Price(INR)'].mean().sort_values(ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=city_avg.values, y=city_avg.index, palette='rocket', ax=ax)
                ax.set_xlabel('Average Price (INR)')
                ax.set_title('Average Price by City (Top 10)')
                st.pyplot(fig)
        
        with tab4:
            year_avg = df_clean.groupby('Year')['Market_Price(INR)'].mean().sort_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(year_avg.index, year_avg.values, marker='o', linewidth=2, markersize=8, color='#4A90E2')
            ax.set_xlabel('Year')
            ax.set_ylabel('Average Price (INR)')
            ax.set_title('Average Price Trend by Year')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # ============================================
    # DOWNLOAD REPORT PAGE
    # ============================================
    elif page == "📥 Download Report":
        st.subheader("📥 Download Reports & Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Available Reports")
            
            # Full dataset
            csv_full = df_clean.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📄 Download Full Dataset (CSV)",
                csv_full,
                "full_car_dataset.csv",
                "text/csv",
                key='download-csv'
            )
            
            # Model performance
            csv_model = result_df.to_csv().encode('utf-8')
            st.download_button(
                "🤖 Download Model Performance (CSV)",
                csv_model,
                "model_performance.csv",
                "text/csv",
                key='download-model'
            )
            
            # Price summary
            summary_stats = df_clean['Market_Price(INR)'].describe().to_frame()
            csv_summary = summary_stats.to_csv().encode('utf-8')
            st.download_button(
                "📈 Download Price Summary (CSV)",
                csv_summary,
                "price_summary.csv",
                "text/csv",
                key='download-summary'
            )
        
        with col2:
            st.markdown("### 📋 Recent Predictions")
            
            if 'predictions' in st.session_state and len(st.session_state.predictions) > 0:
                pred_df = pd.DataFrame(st.session_state.predictions)
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                
                csv_pred = pred_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Download Predictions History",
                    csv_pred,
                    "prediction_history.csv",
                    "text/csv",
                    key='download-pred'
                )
            else:
                st.info("No predictions made yet!")
        
        st.markdown("---")
        st.success("💡 Tip: You can copy data directly from tables and paste into Excel!")

else:
    st.info("📥 Please upload your dataset to start!")
    
    st.markdown("---")
    st.markdown("### 🎯 Features Available:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **💰 Price Prediction**
        - AI-powered estimation
        - Multiple ML models
        - Real-time car images
        - Auto-fill details
        """)
    
    with col2:
        st.markdown("""
        **📊 Compare Cars**
        - Side-by-side view
        - Visual charts
        - Best value finder
        - Multiple cars
        """)
    
    with col3:
        st.markdown("""
        **🧮 EMI Calculator**
        - Monthly payments
        - Loan breakdown
        - Payment schedule
        - Interest analysis
        """)
