import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go

from WattPredictor.config.config import get_config
from WattPredictor.components.inference.predictor import Predictor
from WattPredictor.utils.api_client import EIAClient, WeatherClient

# Page config
st.set_page_config(
    page_title="WattPredictor - Real-Time",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS
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


def get_current_times():
    """Get current UTC and New York Eastern time using standard pytz."""
    eastern_tz = pytz.timezone('US/Eastern')
    utc_now = datetime.now(pytz.utc)
    ny_now = utc_now.astimezone(eastern_tz)
    return utc_now, ny_now


@st.cache_resource
def load_predictor():
    """Load predictor component"""
    config = get_config()
    try:
        return Predictor(config=config)
    except Exception as e:
        st.error(f"Error loading predictor: {str(e)}")
        st.info("Run training first: python src/WattPredictor/pipeline/training_pipeline.py")
        st.stop()


@st.cache_data(ttl=600)
def fetch_live_electricity_data():
    """Fetch live electricity data from EIA API with 10-minute TTL caching"""
    client = EIAClient()
    end_date = datetime.now(pytz.utc)
    start_date = end_date - timedelta(hours=720)
    raw_df = client.fetch_range(start_date, end_date)
    if raw_df.empty:
        return pd.DataFrame()
    return client.process_dataframe(raw_df)


@st.cache_data(ttl=600)
def fetch_live_weather():
    """Fetch current weather from Open-Meteo API with 10-minute TTL caching"""
    client = WeatherClient()
    return client.fetch_current()


@st.cache_data(ttl=300)
def get_cached_predictions(_predictor_instance):
    """Generate and cache demand predictions with 5-minute TTL caching"""
    return _predictor_instance.predict()


def get_color(val, minval, maxval):
    """Color scaling for map"""
    f = (val - minval) / (maxval - minval) if maxval != minval else 0.5
    return (int(f * 255), int((1 - f) * 200), 50, 200)


# === MAIN APP ===
st.title("⚡ Real-Time Electricity Demand Prediction")

utc_now, ny_now = get_current_times()

st.markdown(f"""
<div class="time-display">
    🗽 <b>New York Time:</b> {ny_now.strftime('%A, %B %d, %Y  •  %I:%M:%S %p')} {ny_now.strftime('%Z')}
</div>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Status")

with st.spinner("Loading prediction engine..."):
    predictor = load_predictor()

with st.spinner("Fetching live weather..."):
    weather = fetch_live_weather()

with st.spinner("Fetching live electricity data from EIA..."):
    try:
        elec_df = fetch_live_electricity_data()
        if elec_df.empty:
            st.warning("⚠️ Live EIA data unavailable (check ELEC_API_KEY). Using preprocessed baseline dataset.")
            config = get_config()
            elec_df = pd.read_csv(config.preprocessed_data_path)
            elec_df["date"] = pd.to_datetime(elec_df["date"])
    except Exception as e:
        st.warning(f"⚠️ API error: {str(e)}. Falling back to cached baseline dataset.")
        config = get_config()
        elec_df = pd.read_csv(config.preprocessed_data_path)
        elec_df["date"] = pd.to_datetime(elec_df["date"])

prediction_time_eastern = ny_now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

st.header(f"🎯 Prediction for: {prediction_time_eastern.strftime('%I:%M %p')} {prediction_time_eastern.strftime('%Z')}")
st.caption(f"{prediction_time_eastern.strftime('%A, %B %d, %Y')}")

with st.spinner("Generating real-time predictions..."):
    predictions_raw = get_cached_predictions(predictor)
    predictions_df = predictions_raw.copy()
    predictions_df["name"] = predictions_df["sub_region_code"].map(lambda x: NYISO_ZONES.get(x, {}).get("name", f"Zone {x}"))
    predictions_df["latitude"] = predictions_df["sub_region_code"].map(lambda x: NYISO_ZONES.get(x, {}).get("lat", 42.0))
    predictions_df["longitude"] = predictions_df["sub_region_code"].map(lambda x: NYISO_ZONES.get(x, {}).get("lon", -75.0))

st.sidebar.success("✅ Predictions ready!")

max_d, min_d = predictions_df["predicted_demand"].max(), predictions_df["predicted_demand"].min()
predictions_df["fill_color"] = predictions_df["predicted_demand"].apply(lambda x: get_color(x, min_d, max_d))
predictions_df["radius"] = predictions_df["predicted_demand"] * 5

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("MAP NYISO Zone Predictions")
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
    st.subheader("BAR Peak Zone")
    max_zone = predictions_df.loc[predictions_df["predicted_demand"].idxmax()]
    st.metric("FACTOR Highest Demand", f"{max_zone['name']}")
    st.metric("POWER Predicted", f"{max_zone['predicted_demand']:,.0f} MW")
    
    st.subheader("NYC Weather")
    st.write(f"**Temperature:** {weather.get('temperature_2m', 'N/A')}°C")
    st.write(f"**Humidity:** {weather.get('relative_humidity_2m', 'N/A')}%")
    st.write(f"**Wind Speed:** {weather.get('wind_speed_10m', 'N/A')} m/s")

st.subheader("DETAILS Zone Details")
sorted_df = predictions_df.sort_values("predicted_demand", ascending=False)

cols = st.columns(3)
eastern_tz = pytz.timezone('US/Eastern')
for idx, row in enumerate(sorted_df.head(6).itertuples()):
    with cols[idx % 3]:
        st.metric(f"FACTOR {row.name}", f"{int(row.predicted_demand)} MW")
        
        zone_hist = elec_df[elec_df["sub_region_code"] == row.sub_region_code].tail(168).copy()
        if len(zone_hist) > 0:
            zone_hist["date_eastern"] = pd.to_datetime(zone_hist["date"]).dt.tz_convert(eastern_tz)
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

st.markdown(
    """
    <meta http-equiv="refresh" content="3600">
    """,
    unsafe_allow_html=True
)
