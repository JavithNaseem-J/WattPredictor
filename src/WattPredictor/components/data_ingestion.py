import os
import sys
import json
import requests
import pandas as pd
from pathlib import Path
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
from WattPredictor import logger
from WattPredictor.utils.exception import CustomException
from WattPredictor.components.feature_store import FeatureStore
from WattPredictor.utils.helpers import create_directories, save_json, load_json
from WattPredictor.entity.config_entity import DataIngestionConfig, FeatureStoreConfig
from dotenv import load_dotenv

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
load_dotenv()

class DataIngestion:
    def __init__(self, config: DataIngestionConfig, feature_store_config: FeatureStoreConfig):

        self.config = config
        self.feature_store_config = feature_store_config
        self.feature_store = FeatureStore(feature_store_config)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)



    def _elec_get_api_url(self, year, month, day):


        return self.config.elec_api, {
            "frequency": "hourly",
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "facets[parent][0]": "NYIS",
            "offset": 0,
            "length": 5000,
            "start": f"{year}-{month:02d}-{day:02d}",
            "end": (datetime(year, month, day) + timedelta(days=1)).strftime("%Y-%m-%d"),
            "api_key": self.config.elec_api_key
        }

    def _wx_get_api_url(self, start_date, end_date):


        return self.config.wx_api, {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": ["temperature_2m", "weather_code", "relative_humidity_2m", "wind_speed_10m"],
            "timeformat": "unixtime",
            "timezone": "America/New_York"
        }

    def _fetch_data(self, data_type, *args):


        try:
            if data_type == "electricity":
                year, month, day = args
                file_path = self.config.elec_raw_data / f"hourly_demand_{year}-{month:02d}-{day:02d}.json"
                if file_path.exists():
                    data = load_json(file_path)
                    if 'response' in data and 'data' in data['response']:
                        return pd.DataFrame(data['response']['data'])
                
                url, params = self._elec_get_api_url(year, month, day)
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                create_directories([self.config.elec_raw_data])
                save_json(file_path, data)
                
                if 'response' in data and 'data' in data['response']:
                    return pd.DataFrame(data['response']['data'])
                
            elif data_type == "weather":
                start_date, end_date = args
                file_path = self.config.wx_raw_data / f"weather_data_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.csv"
                if file_path.exists():
                    return pd.read_csv(file_path)
                
                url, params = self._wx_get_api_url(start_date, end_date)
                responses = self.openmeteo.weather_api(url, params=params)
                response = responses[0]
                
                hourly = response.Hourly()
                hourly_data = {
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
                }
                
                df = pd.DataFrame(data=hourly_data)
                create_directories([self.config.wx_raw_data])
                df.to_csv(file_path, index=False)
                return df
            
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Unexpected error fetching {data_type} data: {e}")
            raise CustomException(e, sys)

    def download(self):

        try:
            start = pd.to_datetime(self.config.start_date, utc=True)
            end = pd.to_datetime(self.config.end_date, utc=True)

            elec_data = []
            current_date = start
            while current_date <= end:
                year, month, day = current_date.year, current_date.month, current_date.day
                df = self._fetch_data("electricity", year, month, day)
                if not df.empty:
                    elec_data.append(df)
                current_date += timedelta(days=1)

            wx_df = self._fetch_data("weather", start, end)
            
            if elec_data and not wx_df.empty:
                elec_df = pd.concat(elec_data, ignore_index=True)
                
                elec_df['date'] = pd.to_datetime(elec_df['period'], utc=True)
                wx_df['date'] = pd.to_datetime(wx_df['date'], utc=True)
                
                combined_df = pd.merge(elec_df, wx_df, on="date", how="inner")
                
                if combined_df.empty:
                    logger.warning("Merged dataset is empty, check data alignment")
                    return pd.DataFrame()
                
                logger.info(f"Merged data shape: {combined_df.shape}")


                combined_df.columns = (
                    combined_df.columns.str.lower()
                    .str.replace("-", "_", regex=False)
                    .str.replace(" ", "_", regex=False)
                    .str.strip()
                )
                
                self.feature_store.create_feature_group(
                    name="elec_wx_demand",
                    df=combined_df,
                    primary_key=["subba"],
                    event_time="date",
                    description="Merged electricity demand and weather data for WattPredictor"
                )
                
                create_directories([self.config.data_file.parent])
                combined_df.to_csv(self.config.data_file, index=False)
                logger.info(f"Dataset saved to {self.config.data_file} and Feature Store")
                
                return combined_df
            
            logger.warning("No data fetched, returning empty DataFrame")
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error during download: {e}")
            raise CustomException(e, sys)