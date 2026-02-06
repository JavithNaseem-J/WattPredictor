import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go
import joblib
import time as time_module

# Page config
st.set_page_config(
    page_title="WattPredictor - Real-Time",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for better visuals
st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #0f3460;
    }
    .time-display {
        font-size: 1.1em;
        padding: 12px 20px;
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        margin-bottom: 15px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

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


def get_eastern_offset():
    """
    Get the current Eastern Time offset accounting for DST.
    EST (Standard): UTC-5 (November - March)
    EDT (Daylight): UTC-4 (March - November)
    """
    utc_now = datetime.utcfromtimestamp(time_module.time())
    # DST in US: Second Sunday of March to First Sunday of November
    year = utc_now.year
    
    # March: Second Sunday
    march_start = datetime(year, 3, 8)  # Earliest possible second Sunday
    while march_start.weekday() != 6:  # Find Sunday
        march_start += timedelta(days=1)
    dst_start = march_start.replace(hour=2)  # 2 AM local
    
    # November: First Sunday
    nov_start = datetime(year, 11, 1)
    while nov_start.weekday() != 6:  # Find Sunday
        nov_start += timedelta(days=1)
    dst_end = nov_start.replace(hour=2)  # 2 AM local
    
    # Check if we're in DST period
    if dst_start <= utc_now < dst_end:
        return -4  # EDT
    else:
        return -5  # EST


def get_current_times():
    """
    Get current UTC and New York Eastern time reliably.
    Uses Unix timestamp which is always UTC.
    Automatically handles DST.
    """
    utc_now = datetime.utcfromtimestamp(time_module.time())
    eastern_offset = get_eastern_offset()
    ny_now = utc_now + timedelta(hours=eastern_offset)
    return utc_now, ny_now


def utc_to_eastern(utc_dt):
    """Convert UTC datetime to Eastern time (DST-aware)"""
    return utc_dt + timedelta(hours=get_eastern_offset())


def eastern_to_utc(eastern_dt):
    """Convert Eastern datetime to UTC (DST-aware)"""
    return eastern_dt + timedelta(hours=-get_eastern_offset())


@st.cache_resource
def load_model():
    """Load trained model"""
    return joblib.load("artifacts/trainer/model.joblib")


def fetch_live_electricity_data():
    """Fetch live electricity data from EIA API"""
    from WattPredictor.utils.api_client import EIAClient
    client = EIAClient()
    end_date = datetime.utcfromtimestamp(time_module.time())
    start_date = end_date - timedelta(hours=720)
    raw_df = client.fetch_range(start_date, end_date)
    if raw_df.empty:
        return pd.DataFrame()
    return client.process_dataframe(raw_df)


def fetch_live_weather():
    """Fetch current weather from Open-Meteo API"""
    from WattPredictor.utils.api_client import WeatherClient
    client = WeatherClient()
    return client.fetch_current()


def build_features(zone_id: int, historical_df: pd.DataFrame, weather: dict, prediction_time_utc: datetime) -> pd.DataFrame:
    """Build features for a single zone for real-time prediction"""
    zone_data = historical_df[historical_df["sub_region_code"] == zone_id].sort_values("date")
    demand_values = zone_data["demand"].tail(N_FEATURES).values
    
    if len(demand_values) < N_FEATURES:
        if len(demand_values) > 0:
            demand_values = np.pad(demand_values, (N_FEATURES - len(demand_values), 0), 'edge')
        else:
            demand_values = np.zeros(N_FEATURES)
    
    features = {}
    for i in range(N_FEATURES):
        features[f"demand_previous_{N_FEATURES-i}_hour"] = demand_values[i]
    
    # Use Eastern time for time-based features (as the model was trained on Eastern time patterns)
    prediction_time_eastern = utc_to_eastern(prediction_time_utc)
    
    features["temperature_2m"] = weather.get("temperature_2m", 10.0)
    features["hour"] = prediction_time_eastern.hour
    features["day_of_week"] = prediction_time_eastern.weekday()
    features["month"] = prediction_time_eastern.month
    features["is_weekend"] = 1 if prediction_time_eastern.weekday() >= 5 else 0
    features["is_holiday"] = 0
    features["average_demand_last_4_weeks"] = np.mean(demand_values) if len(demand_values) > 0 else 0
    
    return pd.DataFrame([features])


def predict_all_zones(model, historical_df: pd.DataFrame, weather: dict, prediction_time_utc: datetime) -> pd.DataFrame:
    """Make predictions for all zones"""
    predictions = []
    
    for zone_id in NYISO_ZONES.keys():
        features = build_features(zone_id, historical_df, weather, prediction_time_utc)
        pred = model.predict(features)[0]
        predictions.append({
            "sub_region_code": zone_id,
            "name": NYISO_ZONES[zone_id]["name"],
            "latitude": NYISO_ZONES[zone_id]["lat"],
            "longitude": NYISO_ZONES[zone_id]["lon"],
            "predicted_demand": round(pred, 0),
            "date": prediction_time_utc
        })
    
    return pd.DataFrame(predictions)


def get_color(val, minval, maxval):
    """Color scaling for map"""
    f = (val - minval) / (maxval - minval) if maxval != minval else 0.5
    return (int(f * 255), int((1-f) * 200), 50, 200)


# === MAIN APP ===
st.title("⚡ Real-Time Electricity Demand Prediction")

# Get current times
utc_now, ny_now = get_current_times()

# Display current NY time
st.markdown(f"""
<div class="time-display">
    🗽 <b>New York Time:</b> {ny_now.strftime('%A, %B %d, %Y  •  %I:%M:%S %p')} EST
</div>
""", unsafe_allow_html=True)

# Sidebar - minimal
st.sidebar.header("⚙️ Status")

# Load model
with st.spinner("Loading model..."):
    model = load_model()

# Fetch LIVE weather
with st.spinner("Fetching live weather..."):
    weather = fetch_live_weather()
    current_temp = weather.get("temperature_2m", 10.0)

# Fetch LIVE electricity data
with st.spinner("Fetching live electricity data from EIA..."):
    try:
        elec_df = fetch_live_electricity_data()
        
        if elec_df.empty:
            st.error("❌ Could not fetch live electricity data. Please check your API key.")
            st.stop()
        
        # Calculate data freshness (for internal use)
        latest_data_time = elec_df['date'].max()
        if hasattr(latest_data_time, 'tzinfo') and latest_data_time.tzinfo is not None:
            latest_data_time = latest_data_time.tz_convert(None)
        latest_data_time_utc = pd.Timestamp(latest_data_time).to_pydatetime().replace(tzinfo=None)
        
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        st.info("💡 Falling back to cached data...")
        elec_df = pd.read_csv("artifacts/engineering/preprocessed.csv")
        elec_df["date"] = pd.to_datetime(elec_df["date"])
        latest_data_time = elec_df["date"].max()
        if hasattr(latest_data_time, 'tz'):
            latest_data_time = latest_data_time.tz_localize(None) if latest_data_time.tz is None else latest_data_time.tz_convert(None)
        latest_data_time_utc = pd.Timestamp(latest_data_time).to_pydatetime().replace(tzinfo=None)
        hours_old = (utc_now - latest_data_time_utc).total_seconds() / 3600
        st.warning(f"⚠️ Using cached data ({hours_old:.1f}h old)")

# Calculate next hour prediction time in Eastern
prediction_time_eastern = ny_now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
# Convert to UTC for internal model use
prediction_time_utc = eastern_to_utc(prediction_time_eastern)

# Display prediction target - ALWAYS in Eastern time
st.header(f"🎯 Prediction for: {prediction_time_eastern.strftime('%I:%M %p')} EST")
st.caption(f"{prediction_time_eastern.strftime('%A, %B %d, %Y')}")

# Make predictions
with st.spinner("Generating real-time predictions..."):
    predictions_df = predict_all_zones(model, elec_df, weather, prediction_time_utc)

st.sidebar.success("✅ Predictions ready!")

# Color scaling for map
max_d, min_d = predictions_df["predicted_demand"].max(), predictions_df["predicted_demand"].min()
predictions_df["fill_color"] = predictions_df["predicted_demand"].apply(lambda x: get_color(x, min_d, max_d))
predictions_df["radius"] = predictions_df["predicted_demand"] * 5

# Main content
col1, col2 = st.columns([2, 1])

with col1:
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
        tooltip={"html": "<b>{name}</b><br/>Predicted: {predicted_demand} MW"}
    ))

with col2:
    # Peak Zone
    st.subheader("📊 Peak Zone")
    max_zone = predictions_df.loc[predictions_df["predicted_demand"].idxmax()]
    st.metric("🏭 Highest Demand", f"{max_zone['name']}")
    st.metric("⚡ Predicted", f"{max_zone['predicted_demand']:,.0f} MW")
    
    # Current conditions
    st.subheader("🌤️ NYC Weather")
    st.write(f"**🌡️ Temperature:** {weather.get('temperature_2m', 'N/A')}°C")
    st.write(f"**💧 Humidity:** {weather.get('relative_humidity_2m', 'N/A')}%")
    st.write(f"**💨 Wind Speed:** {weather.get('wind_speed_10m', 'N/A')} m/s")

# Zone details with charts
st.subheader("📈 Zone Details")
sorted_df = predictions_df.sort_values("predicted_demand", ascending=False)

cols = st.columns(3)
for idx, row in enumerate(sorted_df.head(6).itertuples()):
    with cols[idx % 3]:
        st.metric(f"🏭 {row.name}", f"{int(row.predicted_demand)} MW")
        
        # Historical plot
        zone_hist = elec_df[elec_df["sub_region_code"] == row.sub_region_code].tail(168).copy()
        if len(zone_hist) > 0:
            # Convert historical dates to Eastern Time for display
            zone_hist["date_eastern"] = zone_hist["date"].apply(
                lambda x: x + timedelta(hours=-5) if pd.notna(x) else x
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=zone_hist["date_eastern"], y=zone_hist["demand"],
                mode="lines", name="Historical", line=dict(color="#60efff", width=1)
            ))
            fig.add_trace(go.Scatter(
                x=[prediction_time_eastern], y=[row.predicted_demand],
                mode="markers", name="Prediction",
                marker=dict(color="#ff6b6b", size=12, symbol="star"),
                hovertemplate=f"Prediction<br>{prediction_time_eastern.strftime('%I:%M %p')} EST<br>%{{y:,.0f}} MW<extra></extra>"
            ))
            fig.update_layout(
                height=200, 
                margin=dict(l=0,r=0,t=0,b=0), 
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)

# Auto-refresh every hour using meta refresh (non-blocking)
# This refreshes the page without blocking the UI
st.markdown(
    """
    <meta http-equiv="refresh" content="3600">
    """,
    unsafe_allow_html=True
)
