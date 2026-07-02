import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================================
# 1. PAGE CONFIGURATION & HEADER UI
# =====================================================================
st.set_page_config(
    page_title="Bengaluru House Price Predictor", 
    page_icon="🏠", 
    layout="centered"
)

st.title("🏠 Bengaluru Real Estate Valuation Engine")
st.markdown(
    "Enter the structural characteristics of the property below to generate a "
    "real-time market value estimation powered by Machine Learning."
)
st.write("---")

# =====================================================================
# 2. PATH-RESILIENT ASSET LOADER (STREAMLIT CACHED)
# =====================================================================
@st.cache_resource 
def load_assets():
    # Automatically resolve paths relative to this script file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.abspath(os.path.join(current_dir, '..', 'models', 'bengaluru_rf_model.pkl'))
    data_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'processed', 'bengaluru_house_data_engineered.csv'))
    
    # Load model and extract feature schema columns (the model's DNA blueprint)
    model = joblib.load(model_path)
    model_features = pd.read_csv(data_path, nrows=1).drop(columns=['price']).columns.tolist()
    
    return model, model_features

try:
    model, model_features = load_assets()
except Exception as e:
    st.error("❌ Critical Error: Unable to locate or load background pipeline models.")
    st.info("Ensure paths align with: `models/bengaluru_rf_model.pkl` relative to the root.")
    st.stop()

# =====================================================================
# 3. INTERACTIVE USER INPUT UI FORM
# =====================================================================
st.subheader("Property Specifications")

col1, col2 = st.columns(2)

with col1:
    total_sqft = st.number_input("Total Built-up Area (Sq. Ft.)", min_value=300, max_value=10000, value=1200, step=50)
    bhk = st.slider("Bedrooms (BHK Count)", min_value=1, max_value=6, value=2)

with col2:
    bath = st.slider("Bathrooms Count", min_value=1, max_value=5, value=2)
    house_age = st.number_input("Property Age (Years Since Construction)", min_value=0, max_value=50, value=2, step=1)

city_score = st.slider("City Center Proximity Score (1 = Far, 10 = Prime Location)", min_value=1, max_value=10, value=5)

# Extract and isolate location feature tokens from encoded matrices
structural_cols = {'total_sqft', 'bhk', 'bath', 'house_age', 'city_center_proximity_score', 'Unnamed: 0'}
location_columns = [col for col in model_features if col not in structural_cols]
location_columns.sort()

# Parse clean readable display labels for the UI dropdown selection box
clean_ui_names = [col.split('_', 1)[1] if '_' in col else col for col in location_columns]
location_mapping = dict(zip(clean_ui_names, location_columns))

# Fallback protection check
if not clean_ui_names:
    clean_ui_names = ["Default / Global Average Zone"]
    location_mapping = {"Default / Global Average Zone": ""}

selected_ui_location = st.selectbox("Select Neighborhood Location", clean_ui_names)
st.write("---")

# =====================================================================
# 4. INFERENCE ENGINE & VALUE GENERATION
# =====================================================================
if st.button("🔮 Calculate Estimated Market Value", type="primary"):
    
    # Instantiate zero-matrix structure matching model parameters exactly
    input_data = pd.DataFrame(0, index=[0], columns=model_features)
    
    # Map raw numerical structural inputs
    for num_col, val in [('total_sqft', total_sqft), ('bhk', bhk), ('bath', bath), 
                         ('house_age', house_age), ('city_center_proximity_score', city_score)]:
        if num_col in input_data.columns:
            input_data[num_col] = val
            
    # Active the chosen location categorical binary flag to 1
    target_location_col = location_mapping.get(selected_ui_location, "")
    if target_location_col in input_data.columns:
        input_data[target_location_col] = 1
        
    try:
        # Re-enforce explicit order alignment of the original data frame columns
        input_data = input_data[model_features]
        
        # Calculate real-time model prediction values
        prediction = model.predict(input_data)[0]
        
        # Output Presentation Interface Containers
        st.balloons()
        st.success(f"### 🎯 Estimated Property Valuation: **₹{prediction:.2f} Lakhs**")
        
        if prediction >= 100:
            crores = prediction / 100
            st.info(f"💡 Equivalent Valuation: **₹{crores:.2f} Crores**")
            
    except Exception as prediction_error:
        st.error("🚨 Inference alignment anomaly caught during active execution.")
        st.code(f"System Log Details: {prediction_error}")