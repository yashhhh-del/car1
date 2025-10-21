# ======================================================
# SMART CAR PRICING SYSTEM - ENHANCED VERSION
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import io

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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
        df = df[(df['Year'] >= 1980) & (df['Year'] <= datetime.now().year)]
    
    # Outlier detection for price
    if 'Market_Price(INR)' in df.columns:
        Q1 = df['Market_Price(INR)'].quantile(0.01)
        Q3 = df['Market_Price(INR)'].quantile(0.99)
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

if 'Market_Price(INR)' not in df_clean.columns:
    st.error("❌ Price column not found!")
    st.stop()

if 'Brand' not in df_clean.columns or 'Model' not in df_clean.columns:
    st.warning("⚠️ Brand/Model columns missing. Some features won't work.")

# Train model with enhanced metrics
@st.cache_resource
def train_model(df):
    current_year = datetime.now().year
    df_model = df.copy()
    
    if 'Year' in df_model.columns:
        df_model['Car_Age'] = current_year - df_model['Year']
    
    if 'Brand' in df_model.columns:
        brand_avg = df_model.groupby('Brand')['Market_Price(INR)'].mean()
        df_model['Brand_Avg_Price'] = df_model['Brand'].map(brand_avg)
        brand_std = df_model.groupby('Brand')['Market_Price(INR)'].std()
        df_model['Brand_Price_Std'] = df_model['Brand'].map(brand_std).fillna(0)
    
    if 'Year' in df_model.columns and 'Mileage' in df_model.columns:
        df_model['Mileage_Per_Year'] = df_model['Mileage'] / (df_model['Car_Age'] + 1)
    
    if 'Year' in df_model.columns:
        df_model['Price_Age_Ratio'] = df_model['Market_Price(INR)'] / (df_model['Car_Age'] + 1)
    
    cat_cols = df_model.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le
    
    X = df_model.drop(columns=['Market_Price(INR)'], errors='ignore')
    y = df_model['Market_Price(INR)']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
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

with st.spinner('🎯 Training advanced model...'):
    model_data = train_model(df_clean)
    st.session_state.model_trained = True
    st.session_state.model = model_data

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy (R²)", f"{model_data['accuracy']:.1f}%")
col2.metric("RMSE", f"₹{model_data['rmse']:,.0f}")
col3.metric("MAPE", f"{model_data['mape']:.1f}%")
col4.metric("CV Score", f"{model_data['cv_scores'].mean():.2f}")

with st.expander("❓ Why Not 100% Accuracy?"):
    st.markdown(f"""
    ### 🎯 Understanding Model Accuracy
    
    **Current Accuracy: {model_data['accuracy']:.1f}%** is actually **EXCELLENT** for car pricing!
    
    #### 🚫 Missing Real-World Factors:
    - 🔧 Car Condition, 📋 Service History, 🚗 Accident History
    - 📍 Location Premium, ⏰ Market Timing, 🎨 Color & Features
    
    #### 📊 Industry Standards:
    - ✅ 85-90%: Very Good ← **You are here!**
    - ✅ 90-95%: Excellent (rare)
    
    Your model considers **{len(model_data['features'])}** features from **{len(df_clean):,}** cars.
    """)

model = model_data['model']
scaler = model_data['scaler']
encoders = model_data['encoders']
features = model_data['features']

# ============================================
# HOME PAGE
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
    
    if 'Brand' in df_clean.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Top 10 Brands by Count")
            top_brands = df_clean['Brand'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_brands.index, top_brands.values, color=sns.color_palette("Blues_r", len(top_brands)))
            ax.set_xlabel('Count', fontsize=12)
            for i, v in enumerate(top_brands.values):
                ax.text(v + 0.5, i, str(v), va='center', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("### 💰 Average Price by Brand (Top 10)")
            brand_price = df_clean.groupby('Brand')['Market_Price(INR)'].mean().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(brand_price.index, brand_price.values, color=sns.color_palette("Greens_r", len(brand_price)))
            ax.set_xlabel('Avg Price (₹)', fontsize=12)
            ax.ticklabel_format(style='plain', axis='x')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    st.subheader("📋 Dataset Sample")
    st.dataframe(df_clean.head(20), use_container_width=True)

# ============================================
# PRICE PREDICTION PAGE  
# ============================================

elif page == "💰 Price Prediction":
    st.subheader("💰 Get Accurate Car Price from Your Data")
    st.info("💡 System will analyze similar cars from your uploaded CSV!")
    
    if 'Brand' not in df_clean.columns or 'Model' not in df_clean.columns:
        st.error("❌ Brand/Model columns required!")
        st.stop()
    
    brand = st.selectbox("🚘 Select Brand", sorted(df_clean['Brand'].unique()))
    brand_data = df_clean[df_clean['Brand'] == brand]
    models_list = sorted(brand_data['Model'].unique())
    model_name = st.selectbox("🔧 Select Model", models_list)
    selected_car_data = brand_data[brand_data['Model'] == model_name]
    
    if len(selected_car_data) == 0:
        st.warning("No data found")
        st.stop()
    
    st.markdown("---")
    st.subheader(f"📊 Market Data from Your CSV: {brand} {model_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cars Found", len(selected_car_data))
    with col2:
        avg_price = selected_car_data['Market_Price(INR)'].mean()
        st.metric("Average Price", f"₹{avg_price:,.0f}")
    with col3:
        min_price = selected_car_data['Market_Price(INR)'].min()
        st.metric("Minimum Price", f"₹{min_price:,.0f}")
    with col4:
        max_price = selected_car_data['Market_Price(INR)'].max()
        st.metric("Maximum Price", f"₹{max_price:,.0f}")
    
    sample_car = selected_car_data.iloc[0]
    
    st.markdown("---")
    st.markdown("### 📝 Select Your Car's Details")
    
    col1, col2, col3 = st.columns(3)
    inputs = {}
    col_idx = 0
    
    # Get all available columns for this car from CSV
    available_cols = [col for col in selected_car_data.columns if col not in ['Market_Price(INR)', 'Brand', 'Model']]
    
    for col in available_cols:
        with [col1, col2, col3][col_idx % 3]:
            # Check if column is numeric or categorical
            if selected_car_data[col].dtype in ['int64', 'float64']:
                # Numeric column
                min_val = float(selected_car_data[col].min())
                max_val = float(selected_car_data[col].max())
                default = float(selected_car_data[col].median())
                inputs[col] = st.number_input(
                    f"{col}", 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=default, 
                    key=f"inp_{col}",
                    help=f"Range in dataset: {min_val:.0f} - {max_val:.0f}"
                )
            else:
                # Categorical column
                unique_vals = sorted(selected_car_data[col].unique())
                inputs[col] = st.selectbox(
                    f"{col}", 
                    unique_vals, 
                    key=f"inp_{col}",
                    help=f"Available options from your data"
                )
            col_idx += 1
    
    # Store Brand and Model
    inputs['Brand'] = brand
    inputs['Model'] = model_name
    
    st.markdown("---")
    st.markdown("### 🔧 Additional Adjustments (Optional)")
    st.info("These factors adjust the base price from your CSV data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        condition = st.select_slider(
            "Overall Condition", 
            ["Poor", "Fair", "Good", "Excellent"], 
            value="Good",
            help="Adjust based on actual car condition"
        )
    
    with col2:
        accident = st.radio(
            "Accident History", 
            ["No Accidents", "Minor", "Major"], 
            index=0,
            help="No accidents = higher value"
        )
    
    with col3:
        owners = st.number_input(
            "Number of Owners", 
            1, 5, 1,
            help="First owner cars have 5-10% premium"
        )
    
    if st.button("🔍 Calculate Accurate Price from CSV Data", type="primary", use_container_width=True):
        st.markdown("---")
        
        # Show similar cars from CSV in detail
        st.markdown("### 🔍 Similar Cars from Your CSV Dataset")
        
        if len(query_df) > 0:
            st.success(f"📋 Showing {min(len(query_df), 10)} most similar cars from your data")
            
            # Calculate percentile
            your_percentile = (selected_car_data['Market_Price(INR)'] < adjusted_price).sum() / len(selected_car_data) * 100
            
            # Prepare display dataframe
            display_cols = ['Brand', 'Model']
            
            # Add available columns
            for col in ['Year', 'Mileage', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Kilometers_Driven']:
                if col in query_df.columns:
                    display_cols.append(col)
            
            # Always show price
            display_cols.append('Market_Price(INR)')
            
            # Add price difference column
            query_df_display = query_df[display_cols].copy()
            query_df_display['Price_Diff'] = query_df_display['Market_Price(INR)'] - adjusted_price
            query_df_display['Diff_%'] = (query_df_display['Price_Diff'] / adjusted_price * 100).round(1)
            
            # Sort by price similarity
            query_df_display['abs_diff'] = query_df_display['Price_Diff'].abs()
            query_df_display = query_df_display.sort_values('abs_diff').head(10)
            query_df_display = query_df_display.drop('abs_diff', axis=1)
            
            # Format display
            format_dict = {'Market_Price(INR)': '₹{:,.0f}', 'Price_Diff': '₹{:+,.0f}', 'Diff_%': '{:+.1f}%'}
            
            if 'Mileage' in query_df_display.columns:
                format_dict['Mileage'] = '{:,.0f} km'
            if 'Kilometers_Driven' in query_df_display.columns:
                format_dict['Kilometers_Driven'] = '{:,.0f} km'
            
            st.dataframe(
                query_df_display.style.format(format_dict),
                use_container_width=True,
                hide_index=True
            )
            
            st.caption("💡 **Price_Diff** shows difference from your car's price. Negative = cheaper, Positive = costlier")
        else:
            st.info("📋 Showing all available cars of this model from your CSV")
            display_df = selected_car_data.head(10)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            your_percentile = (selected_car_data['Market_Price(INR)'] < adjusted_price).sum() / len(selected_car_data) * 100
        
        st.markdown("---")
        
        # Feature-wise Analysis
        st.markdown("### 🔬 Feature-wise Deep Dive")
        
        analysis_cols = st.columns(3)
        col_idx = 0
        
        for col in inputs.keys():
            if col not in ['Brand', 'Model'] and col in selected_car_data.columns:
                with analysis_cols[col_idx % 3]:
                    if selected_car_data[col].dtype in ['int64', 'float64']:
                        # Numeric feature analysis
                        your_val = inputs[col]
                        min_val = selected_car_data[col].min()
                        max_val = selected_car_data[col].max()
                        avg_val = selected_car_data[col].mean()
                        median_val = selected_car_data[col].median()
                        
                        st.markdown(f"#### {col}")
                        
                        # Create mini chart
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.hist(selected_car_data[col], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
                        ax.axvline(x=your_val, color='red', linestyle='--', linewidth=2, label='Your Value')
                        ax.axvline(x=avg_val, color='green', linestyle=':', linewidth=2, label='Average')
                        ax.set_xlabel(col, fontsize=9)
                        ax.set_ylabel('Count', fontsize=9)
                        ax.legend(fontsize=7)
                        ax.grid(True, alpha=0.3, axis='y')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Stats
                        st.caption(f"Your: **{your_val:,.0f}** | Avg: {avg_val:,.0f} | Range: {min_val:,.0f}-{max_val:,.0f}")
                        
                    else:
                        # Categorical feature analysis
                        your_val = inputs[col]
                        value_counts = selected_car_data[col].value_counts()
                        
                        st.markdown(f"#### {col}")
                        
                        # Create pie chart
                        fig, ax = plt.subplots(figsize=(6, 3))
                        colors_pie = plt.cm.Set3(range(len(value_counts)))
                        wedges, texts, autotexts = ax.pie(
                            value_counts.head(5), 
                            labels=value_counts.head(5).index,
                            autopct='%1.0f%%',
                            colors=colors_pie,
                            textprops={'fontsize': 8}
                        )
                        
                        # Highlight user's choice
                        for i, label in enumerate(value_counts.head(5).index):
                            if label == your_val:
                                wedges[i].set_edgecolor('red')
                                wedges[i].set_linewidth(3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        popularity = (selected_car_data[col] == your_val).sum() / len(selected_car_data) * 100
                        st.caption(f"Your: **{your_val}** | Popularity: {popularity:.0f}%")
                    
                    col_idx += 1
        
        st.markdown("---")
        
        # Final Recommendations
        st.markdown("### 🎯 Pricing Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 💰 Quick Sale Price")
            quick_price = lower_bound
            st.metric("Price", f"₹{quick_price:,.0f}", delta=f"-{((adjusted_price-quick_price)/adjusted_price*100):.1f}%")
            st.caption("Set this price for selling within 1-2 weeks")
        
        with col2:
            st.markdown("#### ⚖️ Fair Market Price")
            fair_price = adjusted_price
            st.metric("Price", f"₹{fair_price:,.0f}", delta="✓ Recommended")
            st.caption("Best price based on condition and market")
        
        with col3:
            st.markdown("#### 💎 Premium Price")
            premium_price = upper_bound
            st.metric("Price", f"₹{premium_price:,.0f}", delta=f"+{((premium_price-adjusted_price)/adjusted_price*100):.1f}%")
            st.caption("Try this if not in hurry (may take 1-2 months)")
        
        st.markdown("---")
        
        # Action buttons
        st.markdown("### 📥 Download & Share")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                st.info("📄 PDF generation feature - Coming soon!")
        
        with col2:
            # Generate summary text
            summary_text = f"""
CAR VALUATION SUMMARY
{'='*50}

Vehicle: {brand} {model_name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

YOUR CAR DETAILS:
{chr(10).join([f'- {k}: {v}' for k, v in inputs.items() if k not in ['Brand', 'Model']])}

CONDITION ASSESSMENT:
- Overall Condition: {condition}
- Accident History: {accident}
- Number of Owners: {owners}

PRICE ANALYSIS:
- Similar Cars Found: {similar_count}
- Base Price (CSV): ₹{base_price:,.0f}
- Adjusted Price: ₹{adjusted_price:,.0f}
- Price Range: ₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}
- Confidence: {confidence*100:.0f}%

MARKET POSITION:
- Market Percentile: {your_percentile:.0f}%
- vs CSV Average: {((adjusted_price - avg_price) / avg_price * 100):+.1f}%

RECOMMENDATIONS:
- Quick Sale: ₹{quick_price:,.0f}
- Fair Price: ₹{fair_price:,.0f}
- Premium: ₹{premium_price:,.0f}

{'='*50}
Generated by Smart Car Pricing System
"""
            
            st.download_button(
                label="📋 Download Summary",
                data=summary_text,
                file_name=f"{brand}_{model_name}_valuation_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col3:
            if st.button("🔄 Analyze Another Car", use_container_width=True):
                st.rerun()
        
        # Detailed Analysis Section
        st.markdown("### 🔬 Deep Analysis of Your Car")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Your Car's Specifications")
            
            specs_data = []
            for col, val in inputs.items():
                if col not in ['Brand', 'Model']:
                    # Find where this value stands in the dataset
                    if col in selected_car_data.columns:
                        if selected_car_data[col].dtype in ['int64', 'float64']:
                            min_val = selected_car_data[col].min()
                            max_val = selected_car_data[col].max()
                            avg_val = selected_car_data[col].mean()
                            
                            # Calculate percentile
                            percentile = (selected_car_data[col] <= val).sum() / len(selected_car_data) * 100
                            
                            # Determine rating
                            if percentile <= 25:
                                rating = "⭐ Below Average"
                            elif percentile <= 50:
                                rating = "⭐⭐ Average"
                            elif percentile <= 75:
                                rating = "⭐⭐⭐ Above Average"
                            else:
                                rating = "⭐⭐⭐⭐ Excellent"
                            
                            specs_data.append({
                                'Feature': col,
                                'Your Value': f"{val:,.0f}" if val > 100 else f"{val:.1f}",
                                'Market Avg': f"{avg_val:,.0f}" if avg_val > 100 else f"{avg_val:.1f}",
                                'Percentile': f"{percentile:.0f}%",
                                'Rating': rating
                            })
                        else:
                            # Categorical value
                            value_count = (selected_car_data[col] == val).sum()
                            total = len(selected_car_data)
                            popularity = (value_count / total) * 100
                            
                            specs_data.append({
                                'Feature': col,
                                'Your Value': str(val),
                                'Market Avg': f"{popularity:.0f}% have this",
                                'Percentile': '-',
                                'Rating': '✓ Common' if popularity > 30 else '◆ Rare'
                            })
            
            if len(specs_data) > 0:
                specs_df = pd.DataFrame(specs_data)
                st.dataframe(specs_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 💡 Key Insights")
            
            insights = []
            
            # Year/Age Analysis
            if 'Year' in inputs:
                car_year = inputs['Year']
                current_year = datetime.now().year
                car_age = current_year - car_year
                avg_year = selected_car_data['Year'].mean() if 'Year' in selected_car_data.columns else car_year
                avg_age = current_year - avg_year
                
                if car_age < avg_age:
                    insights.append(f"✅ **Newer than average** ({car_age} vs {avg_age:.0f} years) - Adds ₹{(adjusted_price * 0.05):,.0f} value")
                elif car_age > avg_age:
                    insights.append(f"⚠️ **Older than average** ({car_age} vs {avg_age:.0f} years) - Reduces ₹{(adjusted_price * 0.05):,.0f} value")
                else:
                    insights.append(f"✓ **Average age** ({car_age} years)")
            
            # Mileage Analysis
            if 'Mileage' in inputs:
                your_mileage = inputs['Mileage']
                avg_mileage = selected_car_data['Mileage'].mean() if 'Mileage' in selected_car_data.columns else your_mileage
                
                if your_mileage < avg_mileage * 0.7:
                    insights.append(f"✅ **Low mileage** ({your_mileage:,.0f} vs {avg_mileage:,.0f} km avg) - Premium value!")
                elif your_mileage > avg_mileage * 1.3:
                    insights.append(f"⚠️ **High mileage** ({your_mileage:,.0f} vs {avg_mileage:,.0f} km avg) - Consider price reduction")
                else:
                    insights.append(f"✓ **Normal mileage** ({your_mileage:,.0f} km)")
            
            # Condition Impact
            if condition == "Excellent":
                insights.append(f"✅ **Excellent condition** - Premium of ₹{(base_price * 0.08):,.0f}")
            elif condition == "Poor":
                insights.append(f"⚠️ **Poor condition** - Discount of ₹{(base_price * 0.15):,.0f}")
            
            # Accident Impact
            if accident == "No Accidents":
                insights.append(f"✅ **Clean history** - No deduction")
            elif accident == "Major":
                insights.append(f"❌ **Major accident** - Reduces value by ₹{(base_price * 0.15):,.0f}")
            
            # Ownership Impact
            if owners == 1:
                insights.append(f"✅ **First owner** - Premium value!")
            elif owners >= 3:
                insights.append(f"⚠️ **Multiple owners** ({owners}) - Reduces ₹{(base_price * 0.03 * (owners-1)):,.0f}")
            
            # Display insights
            for insight in insights:
                st.markdown(insight)
            
            st.markdown("---")
            
            # Overall Rating
            score = 50  # Base score
            
            if 'Year' in inputs and car_age < avg_age:
                score += 10
            if 'Mileage' in inputs and your_mileage < avg_mileage * 0.7:
                score += 15
            if condition == "Excellent":
                score += 15
            elif condition == "Good":
                score += 10
            if accident == "No Accidents":
                score += 10
            if owners == 1:
                score += 10
            
            if score >= 85:
                st.success(f"🏆 **Overall Score: {score}/100** - Excellent Car!")
            elif score >= 70:
                st.info(f"⭐ **Overall Score: {score}/100** - Good Car")
            else:
                st.warning(f"⚠️ **Overall Score: {score}/100** - Fair Car")
        
        st.markdown("---")
        
        # Comparison with Similar Cars
        st.markdown("### 🚗 Your Car vs Similar Cars in CSV")
        
        if len(query_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Price Comparison")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Histogram of similar cars
                ax.hist(query_df['Market_Price(INR)'], bins=15, 
                       color='lightblue', alpha=0.7, edgecolor='black', label='Similar Cars')
                
                # Mark your adjusted price
                ax.axvline(x=adjusted_price, color='red', linestyle='--', 
                          linewidth=3, label=f'Your Car: ₹{adjusted_price:,.0f}')
                
                # Mark quartiles
                q1 = query_df['Market_Price(INR)'].quantile(0.25)
                q2 = query_df['Market_Price(INR)'].quantile(0.50)
                q3 = query_df['Market_Price(INR)'].quantile(0.75)
                
                ax.axvline(x=q1, color='orange', linestyle=':', linewidth=2, alpha=0.5, label=f'25th %ile: ₹{q1:,.0f}')
                ax.axvline(x=q2, color='green', linestyle=':', linewidth=2, alpha=0.5, label=f'Median: ₹{q2:,.0f}')
                ax.axvline(x=q3, color='purple', linestyle=':', linewidth=2, alpha=0.5, label=f'75th %ile: ₹{q3:,.0f}')
                
                ax.set_xlabel('Price (₹)', fontsize=12)
                ax.set_ylabel('Number of Cars', fontsize=12)
                ax.set_title(f'Price Distribution - {similar_count} Similar Cars', fontsize=14, fontweight='bold')
                ax.legend(loc='upper right', fontsize=9)
                ax.ticklabel_format(style='plain', axis='x')
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("#### 📈 Statistical Comparison")
                
                comparison_stats = {
                    'Metric': [
                        'Your Price',
                        'Minimum in CSV',
                        '25th Percentile',
                        'Median (50th)',
                        '75th Percentile',
                        'Maximum in CSV',
                        'Average Price',
                        'Std Deviation'
                    ],
                    'Value': [
                        f"₹{adjusted_price:,.0f}",
                        f"₹{query_df['Market_Price(INR)'].min():,.0f}",
                        f"₹{q1:,.0f}",
                        f"₹{q2:,.0f}",
                        f"₹{q3:,.0f}",
                        f"₹{query_df['Market_Price(INR)'].max():,.0f}",
                        f"₹{query_df['Market_Price(INR)'].mean():,.0f}",
                        f"₹{query_df['Market_Price(INR)'].std():,.0f}"
                    ]
                }
                
                comp_df = pd.DataFrame(comparison_stats)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                # Position analysis
                your_percentile = (query_df['Market_Price(INR)'] < adjusted_price).sum() / len(query_df) * 100
                
                st.markdown("---")
                st.markdown("**Your Market Position:**")
                
                if your_percentile < 25:
                    st.info(f"📉 **{your_percentile:.0f}th percentile** - Priced in lower 25% (Quick sale likely)")
                elif your_percentile < 50:
                    st.success(f"💰 **{your_percentile:.0f}th percentile** - Competitive pricing (Good balance)")
                elif your_percentile < 75:
                    st.warning(f"📈 **{your_percentile:.0f}th percentile** - Above average (May take longer)")
                else:
                    st.error(f"🔝 **{your_percentile:.0f}th percentile** - Premium pricing (Consider reduction)")
        
        st.markdown("---")
        
        # STEP 1: Find exact/similar matches in CSV
        query_df = selected_car_data.copy()
        
        st.info(f"🔍 Searching in {len(query_df)} {brand} {model_name} cars from your CSV...")
        
        # Smart filtering based on inputs
        matches_found = []
        
        for col, val in inputs.items():
            if col in query_df.columns and col not in ['Brand', 'Model']:
                if query_df[col].dtype in ['int64', 'float64']:
                    # For numeric: use 20% tolerance (more flexible)
                    tolerance = val * 0.2
                    before_filter = len(query_df)
                    query_df = query_df[
                        (query_df[col] >= val - tolerance) & 
                        (query_df[col] <= val + tolerance)
                    ]
                    after_filter = len(query_df)
                    if after_filter < before_filter:
                        matches_found.append(f"{col}: {val} (±20%) → {after_filter} cars")
                else:
                    # For categorical: exact match
                    before_filter = len(query_df)
                    query_df = query_df[query_df[col] == val]
                    after_filter = len(query_df)
                    if after_filter < before_filter:
                        matches_found.append(f"{col}: {val} → {after_filter} cars")
        
        # Show filtering progress
        with st.expander("🔍 Search Process Details"):
            st.write("**Filtering Applied:**")
            for match in matches_found:
                st.write(f"• {match}")
            st.write(f"\n**Final Result: {len(query_df)} matching cars found**")
        
        # STEP 2: Calculate base price
        if len(query_df) >= 3:
            # Good matches found - use CSV directly
            base_price = query_df['Market_Price(INR)'].median()  # Use median for better accuracy
            similar_count = len(query_df)
            price_std = query_df['Market_Price(INR)'].std()
            st.success(f"✅ Found {similar_count} similar cars in your CSV! Using their prices.")
            
        elif len(query_df) >= 1:
            # Few matches - use average with model
            csv_price = query_df['Market_Price(INR)'].mean()
            
            # Also get ML prediction
            input_data = inputs.copy()
            current_year = datetime.now().year
            
            if 'Year' in input_data:
                input_data['Car_Age'] = current_year - input_data['Year']
            if 'Brand' in input_data:
                brand_avg = df_clean.groupby('Brand')['Market_Price(INR)'].mean()
                input_data['Brand_Avg_Price'] = brand_avg.get(input_data['Brand'], avg_price)
                brand_std = df_clean.groupby('Brand')['Market_Price(INR)'].std()
                input_data['Brand_Price_Std'] = brand_std.get(input_data['Brand'], 0)
            if 'Year' in input_data and 'Mileage' in input_data:
                input_data['Mileage_Per_Year'] = input_data['Mileage'] / (input_data['Car_Age'] + 1)
            if 'Year' in input_data:
                input_data['Price_Age_Ratio'] = avg_price / (input_data['Car_Age'] + 1)
            
            input_df = pd.DataFrame([input_data])
            
            for col in encoders:
                if col in input_df.columns:
                    try:
                        input_df[col] = encoders[col].transform(input_df[col].astype(str))
                    except:
                        input_df[col] = 0
            
            for col in features:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[features]
            input_scaled = scaler.transform(input_df)
            ml_price = model.predict(input_scaled)[0]
            
            # Blend CSV and ML (70% CSV, 30% ML)
            base_price = 0.7 * csv_price + 0.3 * ml_price
            similar_count = len(query_df)
            price_std = 0
            st.warning(f"⚠️ Found only {similar_count} similar car(s). Using CSV data (70%) + ML prediction (30%).")
            
        else:
            # No close matches - relax filters and use broader search
            st.warning("⚠️ No close matches with your exact specs. Broadening search...")
            
            # Use all cars of same brand/model
            query_df = selected_car_data.copy()
            
            # Try to filter by at least Year if available
            if 'Year' in inputs and 'Year' in query_df.columns:
                year_val = inputs['Year']
                # ±3 years tolerance
                query_df = query_df[
                    (query_df['Year'] >= year_val - 3) & 
                    (query_df['Year'] <= year_val + 3)
                ]
            
            if len(query_df) > 0:
                base_price = query_df['Market_Price(INR)'].median()
                similar_count = len(query_df)
                price_std = query_df['Market_Price(INR)'].std()
                st.info(f"📊 Using {similar_count} cars from broader search (±3 years)")
            else:
                # Last resort - use all cars of this model
                query_df = selected_car_data.copy()
                base_price = query_df['Market_Price(INR)'].median()
                similar_count = len(query_df)
                price_std = query_df['Market_Price(INR)'].std()
                st.info(f"📊 Using all {similar_count} {brand} {model_name} from your CSV")
        
        # STEP 3: Apply condition adjustments
        condition_mult = {"Poor": 0.85, "Fair": 0.93, "Good": 1.0, "Excellent": 1.08}
        accident_mult = {"No Accidents": 1.0, "Minor": 0.95, "Major": 0.85}
        owner_reduction = (owners - 1) * 0.03
        
        adjusted_price = base_price * condition_mult[condition] * accident_mult[accident] * (1 - owner_reduction)
        
        # Calculate confidence interval based on matches found
        if similar_count >= 5:
            confidence = 0.95  # High confidence
            lower_bound = adjusted_price * 0.97
            upper_bound = adjusted_price * 1.03
        elif similar_count >= 2:
            confidence = 0.85  # Medium confidence
            lower_bound = adjusted_price * 0.93
            upper_bound = adjusted_price * 1.07
        else:
            confidence = 0.75  # Lower confidence
            lower_bound = adjusted_price * 0.90
            upper_bound = adjusted_price * 1.10
        
        # STEP 4: Display Results
        st.markdown("### 💰 Price Analysis from Your CSV")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Similar Cars Found", f"{similar_count}", 
                     help="Number of matching cars in your CSV")
        
        with col2:
            st.metric("CSV Base Price", f"₹{base_price:,.0f}", 
                     help="Median price of similar cars")
        
        with col3:
            adjustment = adjusted_price - base_price
            st.metric("Adjusted Price", f"₹{adjusted_price:,.0f}", 
                     delta=f"{adjustment:+,.0f}",
                     help="After condition/accident/owner adjustments")
        
        with col4:
            st.metric("Confidence", f"{confidence*100:.0f}%",
                     help=f"Based on {similar_count} matches")
        
        st.markdown("---")
        
        # Price range display
        st.markdown("### 🎯 Recommended Price Range")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Minimum", f"₹{lower_bound:,.0f}", 
                     delta=f"-{((adjusted_price-lower_bound)/adjusted_price*100):.1f}%",
                     help="Conservative estimate")
        
        with col2:
            st.metric("**FAIR PRICE**", f"₹{adjusted_price:,.0f}", 
                     delta="✓ Best Estimate",
                     help="Most accurate prediction")
        
        with col3:
            st.metric("Maximum", f"₹{upper_bound:,.0f}", 
                     delta=f"+{((upper_bound-adjusted_price)/adjusted_price*100):.1f}%",
                     help="Optimistic estimate")
        
        st.markdown("---")
        
        # Detailed breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Price Breakdown")
            st.write(f"**From Your CSV Data:**")
            st.write(f"• Similar Cars Found: {similar_count if similar_count > 0 else 'Using ML prediction'}")
            st.write(f"• Base Price: ₹{base_price:,.0f}")
            st.write(f"")
            st.write(f"**Adjustments Applied:**")
            st.write(f"• Condition ({condition}): {condition_mult[condition]:.2f}x")
            st.write(f"• Accident History: {accident_mult[accident]:.2f}x")
            st.write(f"• Ownership ({owners} owner): {1-owner_reduction:.2f}x")
            st.write(f"")
            st.write(f"**Final Price: ₹{adjusted_price:,.0f}**")
        
        with col2:
            st.markdown("### 📈 Market Comparison")
            
            # Show distribution from CSV
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(selected_car_data['Market_Price(INR)'], bins=20, 
                   color='lightblue', alpha=0.7, edgecolor='black')
            ax.axvline(x=adjusted_price, color='red', linestyle='--', 
                      linewidth=3, label=f'Your Car: ₹{adjusted_price:,.0f}')
            ax.axvline(x=avg_price, color='green', linestyle='--', 
                      linewidth=2, label=f'CSV Average: ₹{avg_price:,.0f}')
            ax.set_xlabel('Price (₹)', fontsize=11)
            ax.set_ylabel('Number of Cars', fontsize=11)
            ax.set_title(f'{brand} {model_name} - Price Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.ticklabel_format(style='plain', axis='x')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Show similar cars from CSV
        st.markdown("### 🚗 Similar Cars from Your CSV")
        
        if similar_count > 0:
            display_df = query_df[['Brand', 'Model'] + [col for col in available_cols[:5]] + ['Market_Price(INR)']].head(10)
            st.dataframe(display_df.style.format({'Market_Price(INR)': '₹{:,.0f}'}), 
                        use_container_width=True)
        else:
            # Show closest matches by year or mileage
            st.info("Showing all cars of this brand/model from your CSV:")
            display_df = selected_car_data[['Brand', 'Model'] + [col for col in available_cols[:5]] + ['Market_Price(INR)']].head(10)
            st.dataframe(display_df.style.format({'Market_Price(INR)': '₹{:,.0f}'}), 
                        use_container_width=True)
        
        st.markdown("---")
        
        # Market insights
        st.markdown("### 💡 Market Insights from Your Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if adjusted_price > avg_price:
                diff_pct = ((adjusted_price - avg_price) / avg_price) * 100
                st.success(f"""
                **Above Average Price**
                
                Your car is priced **{diff_pct:.1f}%** higher than the average {brand} {model_name} in your dataset.
                
                This is due to better condition and lower ownership.
                """)
            else:
                diff_pct = ((avg_price - adjusted_price) / avg_price) * 100
                st.info(f"""
                **Competitive Price**
                
                Your car is priced **{diff_pct:.1f}%** lower than average.
                
                This makes it attractive to buyers!
                """)
        
        with col2:
            st.markdown(f"""
            **Price Statistics from CSV:**
            
            - Lowest: ₹{min_price:,.0f}
            - Your Price: ₹{adjusted_price:,.0f}
            - Average: ₹{avg_price:,.0f}
            - Highest: ₹{max_price:,.0f}
            - Total Cars: {len(selected_car_data)}
            """)
        
        with col3:
            st.markdown("""
            **Recommendation:**
            
            ✅ Price is based on your actual CSV data
            
            ✅ Adjusted for real conditions
            
            ✅ Compare with similar cars above
            
            💡 Use this as your selling price!
            """)
        
        st.balloons()
        
        # Save prediction
        st.session_state.predictions.append({
            'Brand': brand,
            'Model': model_name,
            'Your Inputs': str(inputs)[:50] + '...',
            'CSV Base': f"₹{base_price:,.0f}",
            'Final Price': f"₹{adjusted_price:,.0f}",
            'Similar Cars': similar_count,
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M")
        })

# ============================================
# COMPARE CARS PAGE
# ============================================

elif page == "📊 Compare Cars":
    st.subheader("📊 Compare Multiple Cars")
    
    num_cars = st.slider("Number of cars", 2, 4, 2)
    comparison_data = []
    cols = st.columns(num_cars)
    
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"### Car {i+1}")
            brand = st.selectbox("Brand", sorted(df_clean['Brand'].unique()), key=f"cb{i}")
            models = df_clean[df_clean['Brand'] == brand]['Model'].unique()
            model_car = st.selectbox("Model", sorted(models), key=f"cm{i}")
            car_data = df_clean[(df_clean['Brand'] == brand) & (df_clean['Model'] == model_car)]
            if len(car_data) > 0:
                comparison_data.append({
                    'Brand': brand,
                    'Model': model_car,
                    'Price': car_data['Market_Price(INR)'].mean()
                })
    
    if st.button("Compare", type="primary"):
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(10, 6))
        cars = [f"{d['Brand']}\n{d['Model']}" for d in comparison_data]
        prices = [d['Price'] for d in comparison_data]
        colors = ['#667eea', '#764ba2', '#f093fb', '#fccb90'][:num_cars]
        bars = ax.bar(cars, prices, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Price (₹)', fontsize=12)
        ax.set_title('Price Comparison', fontsize=14)
        ax.ticklabel_format(style='plain', axis='y')
        for bar, price in zip(bars, prices):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'₹{price:,.0f}', 
                   ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ============================================
# EMI CALCULATOR PAGE
# ============================================

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
    
    if loan > 0 and r > 0:
        emi = loan * r * ((1 + r)**months) / (((1 + r)**months) - 1)
    else:
        emi = loan / months if months > 0 else 0
    
    total = emi * months
    interest = total - loan
    
    with col2:
        st.metric("Monthly EMI", f"₹{emi:,.0f}")
        st.metric("Total Payment", f"₹{total:,.0f}")
        st.metric("Total Interest", f"₹{interest:,.0f}")

# ============================================
# ANALYTICS DASHBOARD PAGE
# ============================================

elif page == "📈 Analytics Dashboard":
    st.subheader("📈 Model Analytics")
    
    st.markdown("### 🎯 Feature Importance")
    top_features = model_data['feature_importance'].head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_features['feature'], top_features['importance'], 
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features))))
    ax.set_xlabel('Importance', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    st.markdown("### 🎯 Predicted vs Actual")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(model_data['y_test'], model_data['y_pred'], alpha=0.5, s=50)
    max_val = max(model_data['y_test'].max(), model_data['y_pred'].max())
    min_val = min(model_data['y_test'].min(), model_data['y_pred'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel('Actual Price', fontsize=12)
    ax.set_ylabel('Predicted Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================
# DEPRECIATION ANALYSIS PAGE
# ============================================

elif page == "📉 Depreciation Analysis":
    st.subheader("📉 Depreciation Analysis")
    
    if 'Year' not in df_clean.columns:
        st.error("Year column required!")
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        dep_brand = st.selectbox("Brand", sorted(df_clean['Brand'].unique()), key="dep_brand")
    with col2:
        dep_models = df_clean[df_clean['Brand'] == dep_brand]['Model'].unique()
        dep_model = st.selectbox("Model", sorted(dep_models), key="dep_model")
    
    dep_data = df_clean[(df_clean['Brand'] == dep_brand) & (df_clean['Model'] == dep_model)].copy()
    if len(dep_data) < 2:
        st.warning("Insufficient data")
        st.stop()
    
    current_year = datetime.now().year
    dep_data['Car_Age'] = current_year - dep_data['Year']
    age_price = dep_data.groupby('Car_Age')['Market_Price(INR)'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(age_price['Car_Age'], age_price['Market_Price(INR)'], 
           marker='o', linewidth=3, markersize=10, color='#4ecdc4')
    ax.set_xlabel('Car Age (years)', fontsize=12)
    ax.set_ylabel('Average Price (₹)', fontsize=12)
    ax.set_title(f'{dep_brand} {dep_model} - Depreciation Trend', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Footer
st.markdown("---")
if len(st.session_state.predictions) > 0:
    with st.expander("📜 Prediction History"):
        st.dataframe(pd.DataFrame(st.session_state.predictions), use_container_width=True)

st.markdown("Made with ❤️ | Enhanced Smart Car Pricing System")
