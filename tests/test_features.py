import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from WattPredictor.utils.ts_generator import (
    get_cutoff_indices_features_and_target,
    features_and_target,
    average_demand_last_4_weeks,
    get_pipeline
)


def create_sample_time_series(n_hours=2000, n_zones=3):
    """Create synthetic time-series data for testing."""
    dates = pd.date_range('2025-01-01', periods=n_hours, freq='H')
    
    data = []
    for zone in range(n_zones):
        for date in dates:
            base_demand = 2000 + zone * 200
            hourly_pattern = np.sin(date.hour / 24 * 2 * np.pi) * 500
            weekly_pattern = np.sin(date.dayofweek / 7 * 2 * np.pi) * 200
            noise = np.random.randn() * 50
            
            data.append({
                'date': date,
                'demand': base_demand + hourly_pattern + weekly_pattern + noise,
                'sub_region_code': zone,
                'temperature_2m': 15 + zone + np.random.randn() * 5,
                'hour': date.hour,
                'day_of_week': date.dayofweek,
                'month': date.month,
                'is_weekend': 1 if date.dayofweek >= 5 else 0,
                'is_holiday': 0
            })
    
    return pd.DataFrame(data)


class TestCutoffIndices:
    """Test cutoff index generation."""
    
    def test_basic_indices(self):
        df = pd.DataFrame({'demand': range(1000)})
        indices = get_cutoff_indices_features_and_target(df, input_seq_len=100, step_size=1)
        
        assert len(indices) > 0
        assert all(isinstance(idx, tuple) for idx in indices)
        assert all(len(idx) == 3 for idx in indices)
    
    def test_indices_with_step_size(self):
        df = pd.DataFrame({'demand': range(1000)})
        
        indices_1 = get_cutoff_indices_features_and_target(df, input_seq_len=100, step_size=1)
        indices_10 = get_cutoff_indices_features_and_target(df, input_seq_len=100, step_size=10)
        
        assert len(indices_10) < len(indices_1) / 5
    
    def test_indices_sequential(self):
        df = pd.DataFrame({'demand': range(1000)})
        indices = get_cutoff_indices_features_and_target(df, input_seq_len=100, step_size=1)
        
        for i in range(min(5, len(indices))):
            start, mid, end = indices[i]
            assert start == i
            assert mid == start + 100
            assert end == mid + 1


class TestFeaturesAndTarget:
    """Test main feature generation function."""
    
    def test_basic_feature_generation(self):
        df = create_sample_time_series(n_hours=1500, n_zones=2)
        
        X, y = features_and_target(df, input_seq_len=672, step_size=1)
        
        assert X.shape[0] > 0
        assert X.shape[1] == 672 + 6
        assert len(y) == len(X)
    
    def test_no_missing_values(self):
        df = create_sample_time_series(n_hours=1500, n_zones=2)
        
        X, y = features_and_target(df, input_seq_len=672, step_size=1)
        
        assert not X.isna().any().any()
        assert not y.isna().any()
    
    def test_feature_data_types(self):
        df = create_sample_time_series(n_hours=1500, n_zones=2)
        
        X, y = features_and_target(df, input_seq_len=672, step_size=1)
        
        assert X.select_dtypes(include=[np.number]).shape == X.shape
        assert pd.api.types.is_numeric_dtype(y)
    
    def test_multiple_zones(self):
        n_zones = 5
        df = create_sample_time_series(n_hours=1500, n_zones=n_zones)
        
        X, y = features_and_target(df, input_seq_len=672, step_size=10)
        
        assert len(X) > 100
    
    def test_insufficient_data_raises_error(self):
        df = create_sample_time_series(n_hours=500, n_zones=1)
        
        with pytest.raises(ValueError, match="No valid time-series sequences"):
            features_and_target(df, input_seq_len=672, step_size=1)
    
    def test_missing_required_columns_raises_error(self):
        df = create_sample_time_series(n_hours=1000, n_zones=1)
        df = df.drop(columns=['demand'])
        
        with pytest.raises(ValueError, match="missing required columns"):
            features_and_target(df, input_seq_len=672, step_size=1)
    
    def test_input_seq_len_too_small_raises_error(self):
        df = create_sample_time_series(n_hours=1000, n_zones=1)
        
        with pytest.raises(ValueError, match="input_seq_len must be >= 672"):
            features_and_target(df, input_seq_len=100, step_size=1)
    
    def test_feature_column_names(self):
        df = create_sample_time_series(n_hours=1500, n_zones=1)
        
        X, y = features_and_target(df, input_seq_len=672, step_size=10)
        
        assert any('demand_previous_' in col for col in X.columns)
        
        additional_features = ['temperature_2m', 'hour', 'day_of_week', 
                              'month', 'is_weekend', 'is_holiday']
        for feature in additional_features:
            assert feature in X.columns


class TestAverageDemandLast4Weeks:

    def test_average_calculation(self):
        n_features = 672
        data = {f'demand_previous_{i}_hour': [100 + i] for i in range(1, n_features + 1)}
        
        data['demand_previous_168_hour'] = [1000]
        data['demand_previous_336_hour'] = [2000]
        data['demand_previous_504_hour'] = [3000]
        data['demand_previous_672_hour'] = [4000]
        
        X = pd.DataFrame(data)
        
        result = average_demand_last_4_weeks(X)
        
        assert 'average_demand_last_4_weeks' in result.columns
        expected_avg = (1000 + 2000 + 3000 + 4000) / 4
        assert result['average_demand_last_4_weeks'].iloc[0] == expected_avg
    
    def test_missing_required_columns_raises_error(self):
        X = pd.DataFrame({'dummy_column': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            average_demand_last_4_weeks(X)


class TestGetPipeline:

    def test_xgboost_pipeline(self):
        pipeline = get_pipeline(model_type="XGBoost")
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
    
    def test_lightgbm_pipeline(self):
        pipeline = get_pipeline(model_type="LightGBM")
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
    
    def test_invalid_model_type_raises_error(self):
        with pytest.raises(ValueError, match="model_type must be"):
            get_pipeline(model_type="InvalidModel")
    
    def test_pipeline_accepts_hyperparameters(self):
        pipeline = get_pipeline(model_type="XGBoost", n_estimators=50, max_depth=3)
        assert pipeline is not None
    
    def test_pipeline_can_fit_predict(self):
        X = pd.DataFrame(np.random.randn(100, 678))
        X.columns = [f'demand_previous_{i+1}_hour' for i in reversed(range(672))] + \
                    ['temperature_2m', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']
        
        y = pd.Series(np.random.randn(100) * 100 + 2000)
        pipeline = get_pipeline(model_type="XGBoost", n_estimators=10)
        
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
