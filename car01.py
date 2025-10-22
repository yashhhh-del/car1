elif page == "💰 Price Prediction":
    st.subheader("💰 Get Accurate Car Price")
    
    # Step 1: Select Brand
    brand = st.selectbox("🚘 Select Brand", sorted(df_clean['Brand'].unique()))
    brand_data = df_clean[df_clean['Brand'] == brand]
    
    # Step 2: Select Model
    model_name = st.selectbox("🔧 Select Model", sorted(brand_data['Model'].unique()))
    selected_car_data = brand_data[brand_data['Model'] == model_name]
    
    if len(selected_car_data) == 0:
        st.warning("No data found for this Brand and Model combination.")
        st.stop()
    
    # Show CSV data for selected Brand + Model
    st.markdown("---")
    st.markdown(f"### 📋 Available {brand} {model_name} in Your CSV")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cars", len(selected_car_data))
    col2.metric("Avg Price", f"₹{selected_car_data['Market_Price(INR)'].mean():,.0f}")
    col3.metric("Min Price", f"₹{selected_car_data['Market_Price(INR)'].min():,.0f}")
    col4.metric("Max Price", f"₹{selected_car_data['Market_Price(INR)'].max():,.0f}")
    
    # Display all cars of this brand+model from CSV
    st.markdown("#### 🚗 All Cars from Your CSV:")
    display_cols = [col for col in selected_car_data.columns if col != 'Market_Price(INR)']
    display_cols.append('Market_Price(INR)')
    st.dataframe(
        selected_car_data[display_cols].style.format({'Market_Price(INR)': '₹{:,.0f}'}),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Select a Specific Car OR Enter Custom Details")
    
    # Option to select existing car or enter custom
    selection_mode = st.radio(
        "Choose Mode:",
        ["📋 Select from CSV", "✏️ Enter Custom Details"],
        horizontal=True
    )
    
    available_cols = [col for col in selected_car_data.columns if col not in ['Market_Price(INR)', 'Brand', 'Model']]
    inputs = {'Brand': brand, 'Model': model_name}
    
    # Initialize car_year based on selection mode
    car_year_input = datetime.now().year # Default for custom input

    if selection_mode == "📋 Select from CSV":
        # User selects an existing car from dropdown
        st.info("💡 Select a car from your CSV data. All details will auto-fill!")
        
        car_options = []
        for idx, row in selected_car_data.iterrows():
            option_text = f"{brand} {model_name}"
            for col in available_cols: # Use all available cols for options
                if col in row:
                    option_text += f" | {col}: {row[col]}"
            option_text += f" | Price: ₹{row['Market_Price(INR)']:,.0f}"
            car_options.append((idx, option_text)) # Store index with option text
        
        selected_option_idx = st.selectbox(
            "🚗 Select Car from CSV:",
            range(len(car_options)),
            format_func=lambda x: car_options[x][1] # Use the text part for display
        )
        
        # Get the original index from selected_option_idx
        original_csv_index = car_options[selected_option_idx][0]
        selected_row = selected_car_data.loc[original_csv_index] # Use .loc with original index
        
        st.markdown("---")
        st.success("✅ Car details auto-filled from CSV!")
        
        col1, col2, col3 = st.columns(3)
        col_idx = 0
        
        for col in available_cols:
            with [col1, col2, col3][col_idx % 3]:
                if col == 'Year': # Handle 'Year' specifically
                    car_year_input = int(selected_row['Year'])
                    inputs[col] = st.number_input(
                        f"{col}", 
                        min_value=1980, # Assuming minimum year from data cleaning
                        max_value=datetime.now().year, 
                        value=car_year_input,
                        step=1,
                        key=f"inp_{col}",
                        help="Auto-filled from CSV"
                    )
                elif selected_car_data[col].dtype in ['int64', 'float64']:
                    inputs[col] = st.number_input(
                        f"{col}", 
                        float(selected_car_data[col].min()), 
                        float(selected_car_data[col].max()), 
                        float(selected_row[col]),
                        key=f"inp_{col}",
                        help="Auto-filled from CSV"
                    )
                else:
                    unique_vals = sorted(selected_car_data[col].unique())
                    default_index = unique_vals.index(selected_row[col]) if selected_row[col] in unique_vals else 0
                    inputs[col] = st.selectbox(
                        f"{col}", 
                        unique_vals, 
                        index=default_index,
                        key=f"inp_{col}",
                        help="Auto-filled from CSV"
                    )
                col_idx += 1
        
        csv_base_price = selected_row['Market_Price(INR)']
        st.info(f"📊 **CSV Price for this selected car:** ₹{csv_base_price:,.0f}")
    
    else:  # Enter Custom Details
        st.info("💡 Enter your car's details manually. We'll find similar cars from CSV!")
        
        col1, col2, col3 = st.columns(3)
        col_idx = 0
        
        for col in available_cols:
            with [col1, col2, col3][col_idx % 3]:
                if col == 'Year': # Handle 'Year' specifically
                    min_year = int(selected_car_data['Year'].min()) if 'Year' in selected_car_data.columns else 1980
                    max_year = int(selected_car_data['Year'].max()) if 'Year' in selected_car_data.columns else datetime.now().year
                    car_year_input = st.number_input(
                        f"{col} (Range: {min_year}-{max_year})", 
                        min_value=min_year, 
                        max_value=max_year, 
                        value=max_year, # Default to most recent year for custom input
                        step=1,
                        key=f"inp_{col}",
                        help=f"CSV range: {min_year} to {max_year}"
                    )
                    inputs[col] = car_year_input
                elif selected_car_data[col].dtype in ['int64', 'float64']:
                    min_val = float(selected_car_data[col].min())
                    max_val = float(selected_car_data[col].max())
                    median_val = float(selected_car_data[col].median())
                    inputs[col] = st.number_input(
                        f"{col} (Range: {min_val:.0f}-{max_val:.0f})", 
                        min_val, 
                        max_val, 
                        median_val,
                        key=f"inp_{col}",
                        help=f"CSV range: {min_val:.0f} to {max_val:.0f}"
                    )
                else:
                    unique_vals = sorted(selected_car_data[col].unique())
                    inputs[col] = st.selectbox(
                        f"{col} (Options from CSV)", 
                        unique_vals, 
                        key=f"inp_{col}",
                        help=f"{len(unique_vals)} options available in CSV"
                    )
                col_idx += 1
    
    st.markdown("---")
    st.markdown("### 🔧 Condition Adjustments")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        condition = st.select_slider("Overall Condition", ["Poor", "Fair", "Good", "Excellent"], value="Good")
    with col2:
        accident = st.radio("Accident History", ["No", "Minor", "Major"], index=0)
    with col3:
        owners = st.number_input("Number of Owners", 1, 5, 1)
    
    # Ensure car_year is properly set from inputs, defaulting to current year if not found
    car_year = inputs.get('Year', datetime.now().year) 
    
    if st.button("🔍 Calculate Final Price", type="primary", use_container_width=True):
        
        # --- (Existing Logic for finding similar cars from CSV to get base_price) ---
        query_df = selected_car_data.copy()
        
        for col, val in inputs.items():
            if col in query_df.columns and col not in ['Brand', 'Model']:
                if query_df[col].dtype in ['int64', 'float64']:
                    # Use a wider range for similarity for base_price calculation
                    query_df = query_df[(query_df[col] >= val * 0.9) & (query_df[col] <= val * 1.1)] 
                else:
                    query_df = query_df[query_df[col] == val]
        
        if len(query_df) >= 2:
            base_price = query_df['Market_Price(INR)'].median()
            similar_count = len(query_df)
        else:
            # Fallback to overall model median if not enough similar cars
            base_price = selected_car_data['Market_Price(INR)'].median()
            similar_count = len(selected_car_data)
        # --- (End Existing Logic) ---

        st.markdown("---")
        st.markdown("### 🌐 Estimated New Car Price (Based on Market Trends)")
        
        with st.spinner('🎯 Estimating new car market price...'):
            current_year = datetime.now().year
            car_age = current_year - car_year

            # --- Enhanced estimated_original calculation ---
            # Try to find the highest price for a new/very new model in the dataset
            new_car_price_candidates = df_clean[
                (df_clean['Brand'] == brand) & 
                (df_clean['Model'] == model_name) & 
                (df_clean['Year'] >= current_year - 1) # Look for cars from current or last year
            ]['Market_Price(INR)']

            if not new_car_price_candidates.empty:
                estimated_original = new_car_price_candidates.max()
                st.info(f"💡 Found a new/recent {brand} {model_name} in your CSV data for estimation.")
            else:
                # If no new cars, use the depreciation-based estimation
                if car_age == 0:
                    # For a brand new car, use the base price as a starting point
                    estimated_original = base_price * 1.2 # Assume a new car is significantly more than a used car median
                elif car_age == 1:
                    estimated_original = base_price / 0.85
                else:
                    estimated_original = base_price / (0.85 * (0.90 ** (car_age - 1)))
                st.info("💡 Estimating new car price based on depreciation from similar used cars.")
            
            # Ensure estimated_original is at least as high as adjusted_price
            if estimated_original < base_price:
                 estimated_original = base_price * 1.1 # Must be higher than used car base

            # Calculate market price ranges (simulating web search results based on this enhanced estimation)
            import random
            random.seed(hash(f"{brand}{model_name}{car_year}")) # Consistent results for same inputs

            web_price_min = estimated_original * random.uniform(0.95, 0.98)  # Lower range for ex-showroom
            web_price_mid = estimated_original * random.uniform(0.99, 1.01)  # Average price
            web_price_max = estimated_original * random.uniform(1.02, 1.05)  # Higher for on-road/premium
            
            # Ensure price ranges are sensible (min < mid < max)
            web_price_min = min(web_price_min, web_price_mid * 0.98)
            web_price_max = max(web_price_max, web_price_mid * 1.02)
            
            st.success(f"✅ Estimated new car market prices for {car_year} {brand} {model_name}!")
            
            # Display web prices (text changed)
            st.markdown("#### 📈 Estimated New Car Price Range")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🔻 Min Price",
                    f"₹{web_price_min:,.0f}",
                    help="Estimated lowest market price (ex-showroom)"
                )
            
            with col2:
                st.metric(
                    "🎯 Avg Price",
                    f"₹{web_price_mid:,.0f}",
                    help="Estimated average market price"
                )
            
            with col3:
                st.metric(
                    "🔺 Max Price",
                    f"₹{web_price_max:,.0f}",
                    help="Estimated highest market price (on-road, premium variant)"
                )
            
            with st.expander("📋 View Estimation Details"):
                st.markdown(f"""
                **Estimation Factors Used:**
                - Market trends from your uploaded CSV data.
                - Average depreciation rates for car age.
                - Year: {car_year}
                - Car Age: {car_age} years
                
                *Note: These prices are estimates based on available data and general market principles, not direct live web scraping.*
                
                **Simulated Query for Context:** `{brand} {model_name} {car_year} new car price India`
                """)
        
        # Apply condition adjustments
        condition_mult = {"Poor": 0.85, "Fair": 0.93, "Good": 1.0, "Excellent": 1.08}
        accident_mult = {"No": 1.0, "Minor": 0.95, "Major": 0.85}
        # Ensure 'owners' is at least 1 for the multiplication
        owners_factor = (1 - (max(1, owners) - 1) * 0.03) 
        adjusted_price = base_price * condition_mult[condition] * accident_mult[accident] * owners_factor
        
        # Ensure adjusted price doesn't exceed new car price significantly
        if adjusted_price > web_price_mid * 1.05: # If used car price is too close/above new
            adjusted_price = web_price_mid * 0.95 # Cap it at a reasonable discount
        
        lower_bound = adjusted_price * 0.95
        upper_bound = adjusted_price * 1.05
        
        # Calculate depreciation (using the new estimated_original)
        depreciation_amount = estimated_original - adjusted_price
        depreciation_percent = (depreciation_amount / estimated_original * 100) if estimated_original > 0 else 0
        
        st.markdown("---")
        st.success("✅ Complete Analysis Ready!")
        
        # (Rest of your code remains the same from here, using the updated estimated_original, web_price_min/mid/max, adjusted_price, etc.)
        # ... your existing comparison metrics and charts ...

        # Price comparison: Original vs Current
        st.markdown("### 💰 Detailed Price Analysis")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "🆕 Original Est. Price", # Changed label
                f"₹{estimated_original:,.0f}",
                help=f"Estimated original new car price for {car_year} model"
            )
        
        with col2:
            st.metric(
                "📉 Depreciation",
                f"₹{depreciation_amount:,.0f}",
                delta=f"-{depreciation_percent:.1f}%",
                delta_color="inverse",
                help="Total estimated value lost since new"
            )
        
        with col3:
            st.metric(
                "📊 CSV Base",
                f"₹{base_price:,.0f}",
                help=f"Median price of {similar_count} similar cars in your data"
            )
        
        with col4:
            st.metric(
                "🎯 Current Value",
                f"₹{adjusted_price:,.0f}",
                help="After condition and history adjustments"
            )
        
        with col5:
            st.metric(
                "📈 Value Retained",
                f"{100-depreciation_percent:.1f}%",
                help="Percentage of estimated original value retained"
            )
        
        st.markdown("---")
        
        # Detailed comparison (Chart)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Price Comparison Chart")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            categories = ['Est. Original\nPrice (New)', 'Current Market\nValue (CSV)', 'Your Car\n(Adjusted)', 'Est. Depreciation\nAmount'] # Changed labels
            values = [estimated_original, base_price, adjusted_price, depreciation_amount]
            colors = ['#4ecdc4', '#667eea', '#f093fb', '#ff6b6b']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'₹{val:,.0f}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
            
            ax.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
            ax.set_title(f'{brand} {model_name} ({car_year}) - Price Analysis', 
                        fontsize=14, fontweight='bold')
            ax.ticklabel_format(style='plain', axis='y')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("### 💡 Detailed Breakdown")
            
            st.markdown(f"""
            **📈 Estimated New Car Prices:**
            - Minimum: ₹{web_price_min:,.0f}
            - Average: ₹{web_price_mid:,.0f}
            - Maximum: ₹{web_price_max:,.0f}
            - Est. Price Range: ₹{web_price_max - web_price_min:,.0f}
            
            **🆕 Your Car Details:**
            - Brand: {brand}
            - Model: {model_name}
            - Year: {car_year}
            - Age: {car_age} years
            - Estimated New Price: ₹{estimated_original:,.0f}
            
            **📊 Market Analysis (from CSV):**
            - Similar Cars in CSV: {similar_count}
            - CSV Base Price: ₹{base_price:,.0f}
            - Market Position: {'Below Average' if adjusted_price < base_price else 'Above Average'}
            
            **🔧 Adjustments Applied:**
            - Condition ({condition}): {condition_mult[condition]:.0%}
            - Accident ({accident}): {accident_mult[accident]:.0%}
            - Owners ({owners}): {(1-(max(1, owners)-1)*0.03):.0%}
            
            **💰 Final Valuation:**
            - Your Car Value: ₹{adjusted_price:,.0f}
            - Price Range: ₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}
            - Discount from Estimated New: {((web_price_mid - adjusted_price) / web_price_mid * 100) if web_price_mid > 0 else 0:.1f}%
            - Total Estimated Depreciation: ₹{depreciation_amount:,.0f} ({depreciation_percent:.1f}%)
            - Estimated Value Retained: {100-depreciation_percent:.1f}%
            
            **📈 Investment Analysis:**
            - Yearly Estimated Depreciation: ₹{depreciation_amount/car_age if car_age > 0 else 0:,.0f}
            - Monthly Estimated Value Loss: ₹{depreciation_amount/(car_age*12) if car_age > 0 else 0:,.0f}
            - Savings vs Estimated New: ₹{web_price_mid - adjusted_price:,.0f}
            """)
        
        st.markdown("---")
        
        # Web Price Range Chart (labels changed)
        st.markdown("### 📊 Estimated New Car Price Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Web price range chart
            fig, ax = plt.subplots(figsize=(8, 5))
            
            labels = ['Min\n(Est. Ex-showroom)', 'Average\n(Est. Market)', 'Max\n(Est. On-road)'] # Changed labels
            values = [web_price_min, web_price_mid, web_price_max]
            colors_web = ['#3498db', '#2ecc71', '#e74c3c']
            
            bars = ax.bar(labels, values, color=colors_web, alpha=0.8, edgecolor='black', linewidth=2)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'₹{val:,.0f}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
            
            ax.set_ylabel('Price (₹)', fontsize=11, fontweight='bold')
            ax.set_title(f'Estimated New {brand} {model_name} {car_year} Prices', # Changed title
                        fontsize=12, fontweight='bold')
            ax.ticklabel_format(style='plain', axis='y')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.caption("📈 Prices are estimates based on your data and market models") # Changed caption
        
        with col2:
            # Price difference breakdown
            st.markdown("**💰 Price Breakdown:**")
            
            savings = web_price_mid - adjusted_price
            savings_percent = (savings / web_price_mid * 100) if web_price_mid > 0 else 0
            
            st.write(f"")
            st.write(f"**Estimated New Car (Avg):** ₹{web_price_mid:,.0f}") # Changed label
            st.write(f"**Your Car Value:** ₹{adjusted_price:,.0f}")
            st.write(f"")
            st.success(f"**💰 You Save:** ₹{savings:,.0f}")
            st.info(f"**📉 Discount:** {savings_percent:.1f}%")
            
            st.write(f"")
            st.write(f"**Reason for Discount:**")
            st.write(f"• Age: {car_age} years = {car_age * 12}% avg depreciation (est.)") # Added (est.)
            st.write(f"• Condition: {condition}")
            st.write(f"• Ownership: {owners} owner(s)")
            st.write(f"• Accident: {accident}")
            
            if savings > 0:
                st.success(f"✅ Great deal! You're buying at {savings_percent:.0f}% discount (vs. estimated new car)") # Clarified
            else:
                st.warning("⚠️ Price seems high compared to estimated new car value") # Clarified
        
        st.markdown("---")
        
        # Show matching cars from CSV
        if len(query_df) > 0:
            st.markdown(f"### 🎯 {len(query_df)} Matching Cars from Your CSV")
            display_df = query_df[display_cols].head(10)
            st.dataframe(
                display_df.style.format({'Market_Price(INR)': '₹{:,.0f}'}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No exact matches found. Showing all cars of this model:")
            st.dataframe(
                selected_car_data[display_cols].head(10).style.format({'Market_Price(INR)': '₹{:,.0f}'}),
                use_container_width=True,
                hide_index=True
            )
        
        st.markdown("---")
        
        # Depreciation timeline (labels changed)
        st.markdown("### 📉 Estimated Depreciation Timeline")
        
        years = list(range(car_year, current_year + 1))
        prices = []
        
        # Calculate year-by-year depreciation
        current_value_timeline = estimated_original # Use the new estimated_original for timeline
        for i, year in enumerate(years):
            if i == 0:
                prices.append(current_value_timeline)
            elif i == 1:
                current_value_timeline *= 0.85  # 15% first year
                prices.append(current_value_timeline)
            else:
                current_value_timeline *= 0.90  # 10% subsequent years
                prices.append(current_value_timeline)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(years, prices, marker='o', linewidth=3, markersize=10, 
               color='#667eea', markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=2)
        ax.axhline(y=adjusted_price, color='red', linestyle='--', linewidth=2, label=f'Your Car Value: ₹{adjusted_price:,.0f}')
        ax.fill_between(years, prices, alpha=0.3, color='#667eea')
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value (₹)', fontsize=12, fontweight='bold')
        ax.set_title('Estimated Car Value Depreciation Over Time', fontsize=14, fontweight='bold') # Changed title
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='plain', axis='y')
        
        # Add value labels
        for year, price in zip(years, prices):
            ax.text(year, price, f'₹{price/100000:.1f}L', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 🎯 Smart Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 💰 Quick Sale")
            quick_price = lower_bound
            st.metric("Price", f"₹{quick_price:,.0f}")
            st.caption("Sell within 2 weeks")
            st.progress(0.7)
        
        with col2:
            st.markdown("#### ⚖️ Fair Price")
            st.metric("Price", f"₹{adjusted_price:,.0f}")
            st.caption("Best market value")
            st.progress(1.0)
        
        with col3:
            st.markdown("#### 💎 Premium")
            premium_price = upper_bound
            st.metric("Price", f"₹{premium_price:,.0f}")
            st.caption("Patient sale (1-2 months)")
            st.progress(0.5)
        
        st.balloons()
        
        # Save prediction
        st.session_state.predictions.append({
            'Brand': brand,
            'Model': model_name,
            'Year': car_year,
            'Original Price (Est.)': f"₹{estimated_original:,.0f}", # Changed label
            'Current Value': f"₹{adjusted_price:,.0f}",
            'Depreciation': f"{depreciation_percent:.1f}%",
            'Similar Cars': similar_count,
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M")
        })

elif page == "📊 Compare Cars":
    # ... (no changes needed here) ...
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
    # ... (no changes needed here) ...
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
