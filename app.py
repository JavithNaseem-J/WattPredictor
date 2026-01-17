import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go
import joblib
import requests
import os

st.set_page_config(layout="wide")

# NYISO Zone coordinates
NYISO_ZONES = {
    0: {"name": "West", "lat": 42.8864, "lon": -78.8784},
    1: {"name": "Genesee", "lat": 43.1610, "lon": -77.6109},
    2: {"name": "Central", "lat": 43.0481, "lon": -76.1474},
    3: {"name": "North", "lat": 44.6995, "lon": -73.4525},
    4: {"name": "Mohawk Valley", "lat": 43.1009, "lon": -75.2327},
    5: {"name": "Capital", "lat": 42.6526, "lon": -73.7562},
    6: {"name": "Hudson Valley", "lat": 41.7004, "lon": -73.9210},
    7: {"name": "Millwood", "lat": 41.2048, "lon": -73.8293},
    8: {"name": "Dunwoodie", "lat": 40.9142, "lon": -73.8557},
    9: {"name": "New York City", "lat": 40.7128, "lon": -74.0060},
    10: {"name": "Long Island", "lat": 40.7891, "lon": -73.1350}
}

N_FEATURES = 672  # 28 days * 24 hours

@st.cache_resource
def load_model():
    """Load trained model"""
    return joblib.load("artifacts/trainer/model.joblib")

@st.cache_data(ttl=3600)
def get_current_weather():
    """Fetch current weather from Open-Meteo API"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "current": "temperature_2m",
            "timezone": "America/New_York"
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return data["current"]["temperature_2m"]
    except:
        return 10.0  # Default fallback

@st.cache_data(ttl=300)
def load_historical_demand():
    """Load historical demand data"""
    df = pd.read_csv("artifacts/engineering/preprocessed.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

def build_features(zone_id: int, historical_df: pd.DataFrame, current_temp: float) -> pd.DataFrame:
    """Build features for a single zone for real-time prediction"""
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    # Get last 672 hours of demand for this zone
    zone_data = historical_df[historical_df["sub_region_code"] == zone_id].sort_values("date")
    demand_values = zone_data["demand"].tail(N_FEATURES).values
    
    # Pad if not enough data
    if len(demand_values) < N_FEATURES:
        demand_values = np.pad(demand_values, (N_FEATURES - len(demand_values), 0), 'edge')
    
    # Build lag features
    features = {}
    for i in range(N_FEATURES):
        features[f"demand_previous_{N_FEATURES-i}_hour"] = demand_values[i]
    
    # Time features
    features["temperature_2m"] = current_temp
    features["hour"] = now.hour
    features["day_of_week"] = now.weekday()
    features["month"] = now.month
    features["is_weekend"] = 1 if now.weekday() >= 5 else 0
    features["is_holiday"] = 0
    features["average_demand_last_4_weeks"] = np.mean(demand_values)
    
    return pd.DataFrame([features])

def predict_all_zones(model, historical_df: pd.DataFrame, current_temp: float) -> pd.DataFrame:
    """Make predictions for all zones"""
    predictions = []
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    for zone_id in NYISO_ZONES.keys():
        features = build_features(zone_id, historical_df, current_temp)
        pred = model.predict(features)[0]
        predictions.append({
            "sub_region_code": zone_id,
            "name": NYISO_ZONES[zone_id]["name"],
            "latitude": NYISO_ZONES[zone_id]["lat"],
            "longitude": NYISO_ZONES[zone_id]["lon"],
            "predicted_demand": round(pred, 0),
            "date": now
        })
    
    return pd.DataFrame(predictions)

# === MAIN APP ===
current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
st.title("⚡ Real-Time Electricity Demand Prediction")
st.header(f"🕐 {current_time.strftime('%Y-%m-%d %H:%M')} UTC")

# Sidebar
st.sidebar.header("⚙️ Status")
progress = st.sidebar.progress(0)

# Load resources
with st.spinner("Loading model..."):
    model = load_model()
    progress.progress(25)

with st.spinner("Fetching current weather..."):
    current_temp = get_current_weather()
    st.sidebar.metric("🌡️ NYC Temperature", f"{current_temp}°C")
    progress.progress(50)

with st.spinner("Loading historical data..."):
    historical_df = load_historical_demand()
    latest_data = historical_df["date"].max()
    st.sidebar.info(f"📊 Data up to: {latest_data}")
    progress.progress(75)

# Make real-time predictions
with st.spinner("Generating predictions..."):
    predictions_df = predict_all_zones(model, historical_df, current_temp)
    progress.progress(100)

st.sidebar.success("✅ Predictions ready!")

# Color scaling for map
def get_color(val, minval, maxval):
    f = (val - minval) / (maxval - minval) if maxval != minval else 0.5
    return (int(f * 255), int((1-f) * 200), 50, 200)

max_d, min_d = predictions_df["predicted_demand"].max(), predictions_df["predicted_demand"].min()
predictions_df["fill_color"] = predictions_df["predicted_demand"].apply(lambda x: get_color(x, min_d, max_d))
predictions_df["radius"] = predictions_df["predicted_demand"] * 5

# Map
st.subheader("🗺️ NYISO Zone Predictions")
st.pydeck_chart(pdk.Deck(
    layers=[pdk.Layer(
        "ScatterplotLayer",
        data=predictions_df,
        get_position=["longitude", "latitude"],
        get_radius="radius",
        get_fill_color="fill_color",
        pickable=True
    )],
    initial_view_state=pdk.ViewState(latitude=42.0, longitude=-75.5, zoom=6, pitch=45),
    tooltip={"html": "<b>{name}</b><br/>Demand: {predicted_demand} MW"}
))

# Zone details
st.subheader("📈 Zone Details")
sorted_df = predictions_df.sort_values("predicted_demand", ascending=False)

cols = st.columns(3)
for idx, row in enumerate(sorted_df.head(6).itertuples()):
    with cols[idx % 3]:
        st.metric(f"🏭 {row.name}", f"{int(row.predicted_demand)} MW")
        
        # Historical plot
        zone_hist = historical_df[historical_df["sub_region_code"] == row.sub_region_code].tail(168)
        if len(zone_hist) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=zone_hist["date"], y=zone_hist["demand"],
                mode="lines", name="Historical", line=dict(color="blue", width=1)
            ))
            fig.add_trace(go.Scatter(
                x=[current_time], y=[row.predicted_demand],
                mode="markers", name="Prediction",
                marker=dict(color="red", size=10, symbol="star")
            ))
            fig.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# Refresh button
if st.button("🔄 Refresh Predictions"):
    st.cache_data.clear()
    st.rerun()
