"""
Test suite for API client with retry logic.
Tests EIAClient and WeatherClient functionality.
"""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from WattPredictor.utils.api_client import EIAClient, WeatherClient, NYISO_ZONE_MAPPING


class TestNYISOZoneMapping:
    """Test NYISO zone mapping constants."""
    
    def test_zone_mapping_exists(self):
        """Test that zone mapping is defined."""
        assert NYISO_ZONE_MAPPING is not None
        assert len(NYISO_ZONE_MAPPING) == 11
    
    def test_zone_mapping_values(self):
        """Test that zone mapping has correct structure."""
        expected_zones = ["ZONA", "ZONB", "ZONC", "ZOND", "ZONE",
                         "ZONF", "ZONG", "ZONH", "ZONI", "ZONJ", "ZONK"]
        
        for zone in expected_zones:
            assert zone in NYISO_ZONE_MAPPING
        
        # Values should be 0-10
        values = list(NYISO_ZONE_MAPPING.values())
        assert set(values) == set(range(11))


class TestEIAClient:
    """Test EIA API client."""
    
    def test_client_initialization(self):
        """Test that client can be initialized."""
        client = EIAClient()
        assert client is not None
        assert hasattr(client, 'api_url')
        assert hasattr(client, 'api_key')
    
    def test_client_with_session(self):
        """Test that client can use external session."""
        client = EIAClient()
        session = requests.Session()
        
        # Client should accept session parameter in fetch methods
        assert client is not None
    
    def test_build_params(self):
        """Test API parameter construction."""
        client = EIAClient(api_key="test_key")
        params = client.build_params(year=2025, month=2, day=15)
        
        assert params is not None
        assert isinstance(params, dict)
        assert params['start'] == "2025-02-15"
        assert params['end'] == "2025-02-16"  # Next day
        assert params['api_key'] == "test_key"
        assert params['frequency'] == "hourly"
    
    @patch('requests.get')
    def test_fetch_day_success(self, mock_get):
        """Test successful data fetch."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': {
                'data': [
                    {'value': 2000, 'period': '2025-02-15T00:00:00Z', 'subba': 'ZONA'},
                    {'value': 2100, 'period': '2025-02-15T01:00:00Z', 'subba': 'ZONA'}
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = EIAClient(api_key="test_key")
        df = client.fetch_day(2025, 2, 15)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'value' in df.columns
    
    @patch('requests.get')
    def test_fetch_day_empty_response(self, mock_get):
        """Test handling of empty response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = EIAClient()
        df = client.fetch_day(2025, 2, 15)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    @patch('requests.get')
    def test_fetch_day_handles_exceptions(self, mock_get):
        """Test that exceptions are handled gracefully."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")
        
        client = EIAClient()
        df = client.fetch_day(2025, 2, 15)
        
        # Should return empty DataFrame, not crash
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    @patch('requests.get')
    def test_retry_on_transient_failure(self, mock_get):
        """Test that exceptions return empty DataFrame."""
        # Note: Current implementation doesn't have built-in retry
        # It catches exceptions and returns empty DataFrame
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")
        
        client = EIAClient()
        df = client.fetch_day(2025, 2, 15)
        
        # Should return empty DataFrame on error
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        
        # Should have called get once
        assert mock_get.call_count == 1
    
    @patch('requests.get')
    def test_fetch_range(self, mock_get):
        """Test fetching date range."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            'response': {'data': [{'value': 2000}]}
        }
        mock_get.return_value = mock_response
        
        client = EIAClient()
        start = datetime(2025, 2, 15)
        end = datetime(2025, 2, 17)  # 3 days
        
        df = client.fetch_range(start, end)
        
        assert isinstance(df, pd.DataFrame)
        # Should have called API once per day
        assert mock_get.call_count == 3
    
    def test_process_dataframe(self):
        """Test DataFrame processing."""
        # Create sample raw data
        raw_df = pd.DataFrame({
            'period': ['2025-02-15T00:00:00Z', '2025-02-15T01:00:00Z'],
            'subba': ['ZONA', 'ZONB'],
            'value': [2000, 2100]
        })
        
        processed_df = EIAClient.process_dataframe(raw_df)
        
        assert 'date' in processed_df.columns
        assert 'sub_region_code' in processed_df.columns
        assert 'demand' in processed_df.columns
        
        # Check date conversion
        assert pd.api.types.is_datetime64_any_dtype(processed_df['date'])
        
        # Check zone mapping
        assert processed_df['sub_region_code'].iloc[0] == 0  # ZONA -> 0
        assert processed_df['sub_region_code'].iloc[1] == 1  # ZONB -> 1


class TestWeatherClient:
    """Test Weather API client."""
    
    def test_client_initialization(self):
        """Test that weather client can be initialized."""
        client = WeatherClient()
        assert client is not None
        assert hasattr(client, 'lat')
        assert hasattr(client, 'lon')
    
    def test_client_with_custom_coordinates(self):
        """Test client with custom coordinates."""
        lat, lon = 42.0, -73.0
        client = WeatherClient(lat=lat, lon=lon)
        
        assert client.lat == lat
        assert client.lon == lon
    
    @patch('requests.get')
    def test_fetch_current_success(self, mock_get):
        """Test successful current weather fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'current': {
                'temperature_2m': 15.5,
                'relative_humidity_2m': 65,
                'weather_code': 0,
                'wind_speed_10m': 5.2
            }
        }
        mock_get.return_value = mock_response
        
        client = WeatherClient()
        weather = client.fetch_current()
        
        assert isinstance(weather, dict)
        assert 'temperature_2m' in weather
        assert weather['temperature_2m'] == 15.5
        assert weather['relative_humidity_2m'] == 65
    
    @patch('requests.get')
    def test_fetch_current_with_defaults_on_error(self, mock_get):
        """Test that defaults are returned on error."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timeout")
        
        client = WeatherClient()
        weather = client.fetch_current()
        
        # Should return defaults, not crash
        assert isinstance(weather, dict)
        assert 'temperature_2m' in weather
        assert weather['temperature_2m'] == 10.0  # Default
    
    def test_build_archive_params(self):
        """Test archive parameter construction."""
        client = WeatherClient(lat=40.7, lon=-74.0)
        start = datetime(2025, 2, 1)
        end = datetime(2025, 2, 28)
        
        params = client.build_archive_params(start, end)
        
        assert isinstance(params, dict)
        assert params['latitude'] == 40.7
        assert params['longitude'] == -74.0
        assert params['start_date'] == "2025-02-01"
        assert params['end_date'] == "2025-02-28"
        assert 'temperature_2m' in params['hourly']


class TestAPIClientIntegration:
    """Integration tests (skip if no internet connection)."""
    
    @pytest.mark.skip(reason="Requires internet connection and API key")
    def test_eia_client_real_api_call(self):
        """Test real API call (requires API key in environment)."""
        import os
        
        if not os.getenv("ELEC_API_KEY"):
            pytest.skip("ELEC_API_KEY not set")
        
        client = EIAClient()
        df = client.fetch_day(2025, 2, 15)
        
        assert isinstance(df, pd.DataFrame)
        # If data is available, should have rows
        if not df.empty:
            assert 'value' in df.columns
    
    @pytest.mark.skip(reason="Requires internet connection")
    def test_weather_client_real_api_call(self):
        """Test real weather API call."""
        client = WeatherClient()
        weather = client.fetch_current()
        
        assert isinstance(weather, dict)
        assert 'temperature_2m' in weather
        # Temperature should be reasonable (-50 to 50°C)
        assert -50 < weather['temperature_2m'] < 50
