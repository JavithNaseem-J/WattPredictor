import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import geopandas as gpd
import pydeck as pdk
from WattPredictor.utils.feature import feature_store_instance
from WattPredictor.components.inference.predictor import Predictor
from WattPredictor.entity.config_entity import PredictionConfig
from WattPredictor.config.inference_config import InferenceConfigurationManager
from WattPredictor.utils.logging import logger
from WattPredictor.utils.plot import plot_one_sample
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(layout="wide")

current_date = datetime.now().replace(minute=0, second=0, microsecond=0)
st.title("Electricity Demand Prediction ⚡")
st.header(f"{current_date} UTC")

progress_bar = st.sidebar.header("⚙️ Working Progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 6

# Define NYISO Zones with approximate centers
nyiso_zones = {
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

def create_nyiso_geo_df():
    zones = []
    for zone_id, info in nyiso_zones.items():
        zones.append({
            "sub_region_code": zone_id,
            "name": info["name"],
            "latitude": info["lat"],
            "longitude": info["lon"]
        })
    df = pd.DataFrame(zones)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    return gdf

def load_predictions_from_store(from_date: datetime, to_date: datetime) -> pd.DataFrame:
    fs = feature_store_instance()
    try:
        predictions_fg = fs.feature_store.get_feature_group(
            name="elec_wx_predictions",
            version=2
        )
        if predictions_fg is None:
            raise Exception("Feature group 'elec_wx_predictions' v2 not found.")
        predictions_df = predictions_fg.read()
        predictions_df["date"] = pd.to_datetime(predictions_df["date"]).dt.tz_convert("UTC")
        mask = (predictions_df["date"] >= from_date) & (predictions_df["date"] <= to_date)
        filtered_df = predictions_df[mask]
        if filtered_df.empty:
            for i in range(1, 7):
                fallback_date = to_date - timedelta(hours=i)
                mask = (predictions_df["date"] >= fallback_date) & (predictions_df["date"] <= fallback_date)
                filtered_df = predictions_df[mask]
                if not filtered_df.empty:
                    logger.warning(f"No predictions for {to_date}, using predictions from {fallback_date}")
                    return filtered_df
            raise Exception("No predictions available for the last 6 hours.")
        return filtered_df
    except Exception as e:
        logger.error(f"Failed to load predictions feature group: {str(e)}")
        return pd.DataFrame()

config = InferenceConfigurationManager().get_data_prediction_config()
predictor = Predictor(config=config)

with st.spinner(text="Creating NYISO zones data"):
    geo_df = create_nyiso_geo_df()
    progress_bar.progress(1 / N_STEPS)

with st.spinner(text="Fetching model predictions from the store"):
    predictions_df = load_predictions_from_store(
        from_date=current_date - timedelta(hours=1),
        to_date=current_date
    )
    progress_bar.progress(2 / N_STEPS)

next_hour_predictions_ready = not predictions_df.empty and not predictions_df[predictions_df.date == current_date].empty
if next_hour_predictions_ready:
    predictions_df = predictions_df[predictions_df.date == current_date]
else:
    with st.spinner(text="Computing model predictions"):
        features = predictor._load_batch_features(current_date)
        predictions_df = predictor.predict(save_to_store=True)
        progress_bar.progress(4 / N_STEPS)
    if not predictions_df.empty and not predictions_df[predictions_df.date == current_date].empty:
        predictions_df = predictions_df[predictions_df.date == current_date]
    else:
        st.subheader("⚠️ The most recent data is not yet available. Using last available predictions")
        logger.error("Features are not available for the current hour. Is your feature pipeline up and running? 🤔")

with st.spinner(text="Preparing data to plot"):
    df = pd.merge(
        geo_df,
        predictions_df,
        right_on="sub_region_code",
        left_on="sub_region_code",
        how="inner"
    )
    
    def pseudocolor(val, minval, maxval, startcolor, stopcolor, alpha=255):
        f = float(val - minval) / (maxval - minval)
        rgb = tuple(int(f * (b - a) + a) for a, b in zip(startcolor, stopcolor))
        return rgb + (alpha,)
    
    BLACK, GREEN = (0, 0, 0), (0, 255, 0)
    df["color_scaling"] = df["predicted_demand"]
    max_pred, min_pred = df["color_scaling"].max(), df["color_scaling"].min()
    df["fill_color"] = df["color_scaling"].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
    df["radius"] = df["predicted_demand"] * 5
    progress_bar.progress(5 / N_STEPS)

with st.spinner(text="Generating NYISO Zones Map"):
    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=41.7004,
        longitude=-73.9210,
        zoom=6,
        max_zoom=16,
        pitch=45,
        bearing=0
    )

    geojson = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_radius="radius",
        get_fill_color="fill_color",
        pickable=True
    )

    tooltip = {
        "html": "<b>Zone:</b> [{sub_region_code}] {name} <br /> <b>Predicted demand:</b> {predicted_demand}"
    }

    r = pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )

    st.pydeck_chart(r)
    progress_bar.progress(6 / N_STEPS)

with st.spinner(text="Plotting time-series data"):
    row_indices = np.argsort(df["predicted_demand"].values)[::-1]
    n_to_plot = 6

    for row_id in row_indices[:n_to_plot]:
        location_id = df["sub_region_code"].iloc[row_id]
        location_name = df["name"].iloc[row_id]
        st.header(f"Zone ID: {location_id} - {location_name}")

        prediction = df["predicted_demand"].iloc[row_id]
        st.metric(label="Predicted demand", value=int(prediction))
        
        fig = plot_one_sample(
            example_id=row_id,
            features=features,
            targets=df["predicted_demand"],
            predictions=pd.Series(df["predicted_demand"]),
            display_title=False
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)