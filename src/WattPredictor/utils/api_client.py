import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

load_dotenv()


NYISO_ZONE_MAPPING = {
    "ZONA": 0, "ZONB": 1, "ZONC": 2, "ZOND": 3, "ZONE": 4,
    "ZONF": 5, "ZONG": 6, "ZONH": 7, "ZONI": 8, "ZONJ": 9, "ZONK": 10
}

NYC_LAT = 40.7128
NYC_LON = -74.0060


class EIAClient:
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.api_url = api_url or os.getenv("ELEC_API")
        self.api_key = api_key or os.getenv("ELEC_API_KEY")
    
    def build_params(self, year: int, month: int, day: int) -> Dict[str, Any]:
        return {
            "frequency": "hourly",
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "facets[parent][0]": "NYIS",
            "offset": 0,
            "length": 5000,
            "start": f"{year}-{month:02d}-{day:02d}",
            "end": (datetime(year, month, day) + timedelta(days=1)).strftime("%Y-%m-%d"),
            "api_key": self.api_key
        }
    
    def fetch_day(self, year: int, month: int, day: int, session: Optional[requests.Session] = None,timeout: int = 30) -> pd.DataFrame:

        params = self.build_params(year, month, day)
        req = session or requests
        
        try:
            response = req.get(self.api_url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            
            if 'response' in data and 'data' in data['response']:
                return pd.DataFrame(data['response']['data'])
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Warning: Failed to fetch EIA data for {year}-{month:02d}-{day:02d}: {e}")
            return pd.DataFrame()
    
    def fetch_range(self, start_date: datetime, end_date: datetime,session: Optional[requests.Session] = None) -> pd.DataFrame:

        all_data = []
        current = start_date
        
        while current <= end_date:
            df = self.fetch_day(current.year, current.month, current.day, session)
            if not df.empty:
                all_data.append(df)
            current += timedelta(days=1)
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    
    @staticmethod
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['period'], utc=True)
        df['sub_region_code'] = df['subba'].map(NYISO_ZONE_MAPPING)
        df['demand'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['date', 'sub_region_code'])
        df = df.sort_values(['sub_region_code', 'date'])
        
        return df


class WeatherClient:
    
    # API endpoints
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(self, lat: float = NYC_LAT, lon: float = NYC_LON):
        self.lat = lat
        self.lon = lon
    
    def fetch_current(self, timeout: int = 10) -> Dict[str, Any]:
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": ["temperature_2m", "relative_humidity_2m", "weather_code", "wind_speed_10m"],
            "timezone": "UTC"
        }
        
        try:
            response = requests.get(self.FORECAST_URL, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            
            current = data.get("current", {})
            return {
                "temperature_2m": current.get("temperature_2m", 10.0),
                "relative_humidity_2m": current.get("relative_humidity_2m", 50.0),
                "weather_code": current.get("weather_code", 0),
                "wind_speed_10m": current.get("wind_speed_10m", 5.0)
            }
        except Exception as e:
            print(f"Warning: Failed to fetch current weather: {e}")
            return {
                "temperature_2m": 10.0,
                "relative_humidity_2m": 50.0,
                "weather_code": 0,
                "wind_speed_10m": 5.0
            }
    
    def build_archive_params(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        return {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": ["temperature_2m", "weather_code", "relative_humidity_2m", "wind_speed_10m"],
            "timeformat": "unixtime",
            "timezone": "America/New_York"
        }


# Convenience functions for simple usage
def get_eia_client() -> EIAClient:
    return EIAClient()


def get_weather_client() -> WeatherClient:
    return WeatherClient()
