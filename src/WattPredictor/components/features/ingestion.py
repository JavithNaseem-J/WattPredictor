import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
from dotenv import load_dotenv

from WattPredictor.utils.logging import logger
from WattPredictor.utils.exception import CustomException
from WattPredictor.utils.helpers import create_directories, save_json, load_json
from WattPredictor.utils.api_client import EIAClient, WeatherClient, NYISO_ZONE_MAPPING
from WattPredictor.entity.config_entity import IngestionConfig

# Setup cached session for batch operations
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
load_dotenv()


class Ingestion:
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.eia_client = EIAClient(
            api_url=config.elec_api,
            api_key=config.elec_api_key
        )
        self.weather_client = WeatherClient()
        self.openmeteo = openmeteo_requests.Client(session=retry_session)

    def _fetch_electricity_data(self, year, month, day):
        file_path = self.config.elec_raw_data / f"hourly_demand_{year}-{month:02d}-{day:02d}.json"
        target_date = datetime(year, month, day)
        now = datetime.now()
        
        # Only use cache for data that's at least 2 days old (API may have delays)
        days_old = (now - target_date).days
        use_cache = days_old >= 2
        
        if use_cache and file_path.exists():
            data = load_json(file_path)
            if 'response' in data and 'data' in data['response']:
                return pd.DataFrame(data['response']['data'])

        # Build params using shared client
        params = self.eia_client.build_params(year, month, day)
        
        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = retry_session.get(self.eia_client.api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Save to cache for DVC tracking
                create_directories([self.config.elec_raw_data])
                save_json(file_path, data)
                
                return pd.DataFrame(data['response']['data']) if 'response' in data and 'data' in data['response'] else pd.DataFrame()
            except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {year}-{month:02d}-{day:02d}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch data for {year}-{month:02d}-{day:02d} after {max_retries} attempts")
                    return pd.DataFrame() 
                time.sleep(2 ** attempt)

    def _fetch_weather_chunk(self, url, start_date, end_date):
        params = self.weather_client.build_archive_params(start_date, end_date)
        responses = self.openmeteo.weather_api(url, params=params)
        hourly = responses[0].Hourly()
        return pd.DataFrame({
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "weather_code": hourly.Variables(1).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(2).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(3).ValuesAsNumpy()
        })

    def _fetch_weather_data(self, start_date, end_date):
        file_path = self.config.wx_raw_data / f"weather_data_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.csv"
        
        if file_path.exists():
            return pd.read_csv(file_path)

        archive_url = "https://archive-api.open-meteo.com/v1/archive"
        forecast_url = self.config.wx_api
        # Forecast API covers ~90 days back; use archive for older data
        cutoff = pd.Timestamp.now(tz="UTC") - timedelta(days=85)
        
        dfs = []
        if start_date < cutoff:
            archive_end = min(end_date, cutoff)
            logger.info(f"Fetching archive weather: {start_date.strftime('%Y-%m-%d')} to {archive_end.strftime('%Y-%m-%d')}")
            dfs.append(self._fetch_weather_chunk(archive_url, start_date, archive_end))
        if end_date >= cutoff:
            forecast_start = max(start_date, cutoff)
            logger.info(f"Fetching forecast weather: {forecast_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            dfs.append(self._fetch_weather_chunk(forecast_url, forecast_start, end_date))
        
        df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

        # Save to cache for DVC tracking
        create_directories([self.config.wx_raw_data])
        df.to_csv(file_path, index=False)
        return df

    def _prepare_and_merge(self, elec_data_list, weather_df):
        elec_df = pd.concat(elec_data_list, ignore_index=True)
        elec_df["date"] = pd.to_datetime(elec_df["period"], utc=True)
        weather_df["date"] = pd.to_datetime(weather_df["date"], utc=True)

        combined_df = pd.merge(elec_df, weather_df, on="date", how="inner")

        combined_df.columns = (
            combined_df.columns.str.lower()
            .str.replace("-", "_", regex=False)
            .str.replace(" ", "_", regex=False)
            .str.strip()
        )

        combined_df["date_str"] = combined_df["date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        return combined_df

    def download(self):
        try:
            now = datetime.now().replace(minute=0, second=0, microsecond=0)
            start = now - timedelta(days=365)
            end = now

            start = pd.to_datetime(start, utc=True)
            end = pd.to_datetime(end, utc=True)

            current = start
            elec_data = []

            while current <= end:
                year, month, day = current.year, current.month, current.day
                df = self._fetch_electricity_data(year, month, day)
                if not df.empty:
                    elec_data.append(df)
                current += timedelta(days=1)

            if not elec_data:
                logger.warning("No electricity data fetched.")
                return pd.DataFrame()

            wx_df = self._fetch_weather_data(start, end)
            if wx_df.empty:
                logger.warning("No weather data fetched.")
                return pd.DataFrame()

            combined_df = self._prepare_and_merge(elec_data, wx_df)

            if combined_df.empty:
                logger.warning("Merged DataFrame is empty after join.")
                return pd.DataFrame()

            create_directories([self.config.data_file.parent])
            combined_df.to_csv(self.config.data_file, index=False)
            logger.info(f"Saved combined dataset to {self.config.data_file}")

            return combined_df

        except Exception as e:
            logger.error(f"Error in DataIngestion.download: {str(e)}")
            raise CustomException(e, sys)
