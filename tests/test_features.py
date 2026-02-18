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
from WattPredictor.utils.exception import CustomException


def create_sample_time_series(n_hours=2000, n_zones=3):
    """Create synthetic time-series data for testing."""
    dates = pd.date_range('2025-01-01', periods=n_hours, freq='H')
    
    data = []
    for zone in range(n_zones):
        for date in dates:
            # Create realistic demand pattern
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
        """Test that indices are generated correctly."""
        df = pd.DataFrame({'demand': range(1000)})
        indices = get_cutoff_indices_features_and_target(df, input_seq_len=100, step_size=1)
        
        assert len(indices) > 0
        assert all(isinstance(idx, tuple) for idx in indices)
        assert all(len(idx) == 3 for idx in indices)  # (start, mid, end)
    
    def test_indices_with_step_size(self):
        """Test indices with different step sizes."""
        df = pd.DataFrame({'demand': range(1000)})
        
        # Step size 1
        indices_1 = get_cutoff_indices_features_and_target(df, input_seq_len=100, step_size=1)
        
        # Step size 10
        indices_10 = get_cutoff_indices_features_and_target(df, input_seq_len=100, step_size=10)
        
        # Step size 10 should have ~10x fewer indices
        assert len(indices_10) < len(indices_1) / 5
    
    def test_indices_sequential(self):
        """Test that indices are sequential."""
        df = pd.DataFrame({'demand': range(1000)})
        indices = get_cutoff_indices_features_and_target(df, input_seq_len=100, step_size=1)
        
        # Check first few indices
        for i in range(min(5, len(indices))):
            start, mid, end = indices[i]
            assert start == i
            assert mid == start + 100
            assert end == mid + 1


class TestFeaturesAndTarget:
    """Test main feature generation function."""
    
    def test_basic_feature_generation(self):
        """Test that features are generated correctly."""
        df = create_sample_time_series(n_hours=1500, n_zones=2)
        
        X, y = features_and_target(df, input_seq_len=672, step_size=1)
        
        # Check shapes
        assert X.shape[0] > 0
        assert X.shape[1] == 672 + 6  # 672 lags + 6 additional features
        assert len(y) == len(X)
    
    def test_no_missing_values(self):
        """Test that generated features have no missing values."""
        df = create_sample_time_series(n_hours=1500, n_zones=2)
        
        X, y = features_and_target(df, input_seq_len=672, step_size=1)
        
        assert not X.isna().any().any(), " found NaN values in features"
        assert not y.isna().any(), "Found NaN values in targets"
    
    def test_feature_data_types(self):
        """Test that all features are numeric."""
        df = create_sample_time_series(n_hours=1500, n_zones=2)
        
        X, y = features_and_target(df, input_seq_len=672, step_size=1)
        
        # All features should be numeric
        assert X.select_dtypes(include=[np.number]).shape == X.shape
        assert pd.api.types.is_numeric_dtype(y)
    
    def test_multiple_zones(self):
        """Test that features are generated for multiple zones."""
        n_zones = 5
        df = create_sample_time_series(n_hours=1500, n_zones=n_zones)
        
        X, y = features_and_target(df, input_seq_len=672, step_size=10)  # Use step_size to speed up
        
        # Should have features from multiple zones
        assert len(X) > 100
    
    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises CustomException."""
        df = create_sample_time_series(n_hours=500, n_zones=1)  # Not enough for 672 lags
        
        with pytest.raises(CustomException, match="No valid time-series sequences"):
            features_and_target(df, input_seq_len=672, step_size=1)
    
    def test_missing_required_columns_raises_error(self):
        """Test that missing columns raises CustomException."""
        df = create_sample_time_series(n_hours=1000, n_zones=1)
        df = df.drop(columns=['demand'])  # Remove required column
        
        with pytest.raises(CustomException, match="missing required columns"):
            features_and_target(df, input_seq_len=672, step_size=1)
    
    def test_input_seq_len_too_small_raises_error(self):
        """Test that too-small input_seq_len raises error."""
        df = create_sample_time_series(n_hours=1000, n_zones=1)
        
        with pytest.raises(CustomException, match="input_seq_len must be >= 672"):
            features_and_target(df, input_seq_len=100, step_size=1)
    
    def test_feature_column_names(self):
        """Test that feature columns have correct names."""
        df = create_sample_time_series(n_hours=1500, n_zones=1)
        
        X, y = features_and_target(df, input_seq_len=672, step_size=10)
        
        # Check for lag columns
        assert any('demand_previous_' in col for col in X.columns)
        
        # Check for additional features
        additional_features = ['temperature_2m', 'hour', 'day_of_week', 
                              'month', 'is_weekend', 'is_holiday']
        for feature in additional_features:
            assert feature in X.columns, f"Missing feature: {feature}"


class TestAverageDemandLast4Weeks:
    """Test the average demand calculation."""
    
    def test_average_calculation(self):
        """Test that average is calculated correctly."""
        # Create simple data with one row
        n_features = 672
        
        # Create one row of lag features
        data = {f'demand_previous_{i}_hour': [100 + i] for i in range(1, n_features + 1)}
        
        # Override specific lag columns with known values
        data['demand_previous_168_hour'] = [1000]  # 1 week
        data['demand_previous_336_hour'] = [2000]  # 2 weeks
        data['demand_previous_504_hour'] = [3000]  # 3 weeks
        data['demand_previous_672_hour'] = [4000]  # 4 weeks
        
        X = pd.DataFrame(data)
        
        result = average_demand_last_4_weeks(X)
        
        assert 'average_demand_last_4_weeks' in result.columns
        expected_avg = (1000 + 2000 + 3000 + 4000) / 4
        assert result['average_demand_last_4_weeks'].iloc[0] == expected_avg
    
    def test_missing_required_columns_raises_error(self):
        """Test that missing required columns raises error."""
        X = pd.DataFrame({'dummy_column': [1, 2, 3]})
        
        with pytest.raises(CustomException):
            average_demand_last_4_weeks(X)


class TestGetPipeline:
    """Test pipeline creation."""
    
    def test_xgboost_pipeline(self):
        """Test XGBoost pipeline creation."""
        pipeline = get_pipeline(model_type="XGBoost")
        
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
    
    def test_lightgbm_pipeline(self):
        """Test LightGBM pipeline creation."""
        pipeline = get_pipeline(model_type="LightGBM")
        
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
    
    def test_invalid_model_type_raises_error(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError, match="model_type must be"):
            get_pipeline(model_type="InvalidModel")
    
    def test_pipeline_accepts_hyperparameters(self):
        """Test that pipeline accepts custom hyperparameters."""
        pipeline = get_pipeline(model_type="XGBoost", n_estimators=50, max_depth=3)
        
        assert pipeline is not None
    
    def test_pipeline_can_fit_predict(self):
        """Test that pipeline can fit and predict."""
        # Create simple data
        X = pd.DataFrame(np.random.randn(100, 678))  # 672 lags + 6 features
        
        # Add required columns for average_demand_last_4_weeks
        X.columns = [f'demand_previous_{i+1}_hour' for i in reversed(range(672))] + \
                    ['temperature_2m', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']
        
        y = pd.Series(np.random.randn(100) * 100 + 2000)
        
        pipeline = get_pipeline(model_type="XGBoost", n_estimators=10)
        
        # Should be able to fit
        pipeline.fit(X, y)
        
        # Should be able to predict
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)


class TestFeatureGenerationPerformance:
    """Test feature generation performance."""
    
    def test_generation_completes_in_reasonable_time(self):
        """Test that feature generation is fast enough."""
        import time
        
        df = create_sample_time_series(n_hours=2000, n_zones=5)
        
        start_time = time.time()
        X, y = features_and_target(df, input_seq_len=672, step_size=10)
        elapsed = time.time() - start_time
        
        # Should complete in < 10 seconds with optimization
        assert elapsed < 10, f"Feature generation too slow: {elapsed:.2f}s"
        
        # Should generate features
        assert len(X) > 0
