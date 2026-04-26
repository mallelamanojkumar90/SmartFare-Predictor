import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image

# Page config
st.set_page_config(page_title="SmartFare Dynamic Pricing", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('ride_pricing_app/model/model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please run train.py first.")
    st.stop()

st.title("🚕 Ride Sharing Dynamic Pricing System")
st.markdown("Predict the price multiplier based on real-time demand, supply, and environmental factors.")

# Sidebar Inputs
st.sidebar.header("Ride Parameters")

demand = st.sidebar.slider("Current Demand", 10, 100, 50)
supply = st.sidebar.slider("Available Supply", 10, 100, 50)
zone = st.sidebar.selectbox("Pickup Zone", ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'])
time_of_day = st.sidebar.selectbox("Time of Day", ['morning', 'afternoon', 'evening', 'night'])
day_of_week = st.sidebar.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
weather = st.sidebar.selectbox("Weather", ['sunny', 'rainy', 'cloudy'])
distance = st.sidebar.number_input("Trip Distance (miles)", 0.5, 50.0, 5.0)

# Preprocessing for prediction
# We must match the feature set used in train.py
demand_supply_ratio = demand / supply
is_peak = 1 if time_of_day in ['morning', 'evening'] else 0

input_data = pd.DataFrame([[
    demand, supply, zone, time_of_day, day_of_week, weather, distance, demand_supply_ratio, is_peak
]], columns=[
    'demand', 'supply', 'pickup_zone', 'time_of_day', 'day_of_week', 'weather', 'trip_distance', 'demand_supply_ratio', 'is_peak'
])

# Prediction
prediction = model.predict(input_data)[0]

# UI Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Price Prediction")
    st.metric(label="Predicted Price Multiplier", value=f"{prediction:.2f}x")

    # Interpretation
    if prediction >= 2.0:
        st.warning("🚨 High Surge Pricing Active")
    elif prediction >= 1.3:
        st.info("📈 Moderate Surge Pricing")
    else:
        st.success("✅ Standard Pricing")

with col2:
    st.subheader("Model Insights")
    try:
        img = Image.open('ride_pricing_app/model/feature_importance.png')
        st.image(img, caption="Key Drivers of Pricing", use_column_width=True)
    except FileNotFoundError:
        st.write("Feature importance plot not found. Run train.py to generate it.")

st.divider()
st.markdown("""
### How it works:
- **Demand/Supply Ratio**: The primary driver. High demand with low supply triggers surge.
- **Weather**: Rainy weather increases the multiplier due to higher demand and slower traffic.
- **Peak Hours**: Morning and evening rushes are weighted higher.
- **Location**: Different zones have different base demand patterns.
""")
