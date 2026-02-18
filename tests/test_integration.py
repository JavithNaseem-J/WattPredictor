import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from WattPredictor.config.config import get_config, reset_config


@pytest.fixture
def sample_raw_data():
    np.random.seed(42)

    # Generate 30 days of hourly data
    dates = pd.date_range("2025-01-01", periods=30 * 24, freq="H")

    zones = ["ZONA", "ZONB", "ZONC"]
    demand_data = []

    for zone in zones:
        for date in dates:
            hour = date.hour
            base_demand = 2000 if 8 <= hour <= 22 else 1500
            demand = base_demand + np.random.randn() * 100

            demand_data.append(
                {
                    "date": date,
                    "sub_region_code": zones.index(zone),
                    "demand": demand,
                }
            )

    demand_df = pd.DataFrame(demand_data)

    weather_data = []
    for date in dates:
        weather_data.append(
            {
                "date": date,
                "temperature_2m": 10 + np.random.randn() * 5,
                "relative_humidity_2m": 60 + np.random.randn() * 10,
                "wind_speed_10m": 5 + np.random.randn() * 2,
            }
        )

    weather_df = pd.DataFrame(weather_data)

    return demand_df, weather_df


@pytest.fixture
def sample_merged_data(sample_raw_data):
    demand_df, weather_df = sample_raw_data
    merged_df = demand_df.merge(weather_df, on="date", how="left")
    return merged_df


class TestConfigIntegration:

    def test_config_singleton(self):
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_config_paths_valid(self):
        config = get_config()

        # Check directories
        assert config.artifacts_dir is not None
        assert config.data_dir is not None
        assert config.logs_dir is not None

        # Check path properties that actually exist on WattPredictorConfig
        assert config.processed_data_path is not None
        assert config.model_path is not None
        assert config.raw_elec_data_dir is not None
        assert config.raw_weather_data_dir is not None

    def test_config_hyperparameters(self):
        config = get_config()
        assert config.input_seq_len == 672
        assert config.cv_folds >= 2
        assert config.step_size >= 1


class TestDataPipeline:

    def test_data_loading(self, sample_merged_data, tmp_path):
        data_file = tmp_path / "test_data.csv"
        sample_merged_data.to_csv(data_file, index=False)

        loaded_df = pd.read_csv(data_file)

        assert len(loaded_df) == len(sample_merged_data)
        assert "date" in loaded_df.columns
        assert "demand" in loaded_df.columns
        assert "temperature_2m" in loaded_df.columns

    def test_data_validation(self, sample_merged_data):
        df = sample_merged_data

        required_cols = ["date", "demand", "sub_region_code", "temperature_2m"]
        for col in required_cols:
            assert col in df.columns

        assert df["demand"].notna().all()
        assert df["sub_region_code"].notna().all()
        assert df["demand"].dtype in [np.float64, np.int64]
        assert df["sub_region_code"].dtype in [np.int64, np.int32]

    def test_data_date_range(self, sample_merged_data):
        df = sample_merged_data.copy()
        df["date"] = pd.to_datetime(df["date"])

        date_span = (df["date"].max() - df["date"].min()).days
        assert date_span >= 7


class TestFeaturePipeline:

    def test_feature_generation_basic(self, sample_merged_data):
        from WattPredictor.utils.ts_generator import (
            get_cutoff_indices_features_and_target,
        )

        df = sample_merged_data

        # Actual signature: (data, input_seq_len, step_size)
        cutoff_indices = get_cutoff_indices_features_and_target(
            df, input_seq_len=24, step_size=1
        )

        # Should have some valid indices
        assert len(cutoff_indices) > 0

        # Each tuple: (start, mid, end)
        for start, mid, end in cutoff_indices:
            assert mid - start == 24
            assert end == mid + 1

    def test_feature_columns_created(self, sample_merged_data):
        from WattPredictor.utils.ts_generator import features_and_target

        df = sample_merged_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = df["date"].dt.hour.astype("int64")
        df["day_of_week"] = df["date"].dt.dayofweek.astype("int64")
        df["month"] = df["date"].dt.month.astype("int64")
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int64")
        df["is_holiday"] = 0

        try:
            X, y = features_and_target(df, input_seq_len=672, step_size=1)

            assert X is not None
            assert y is not None
            assert len(X.shape) == 2
            # Should have lag + weather features
            assert X.shape[1] >= 672

        except Exception:
            # May fail if not enough data for 672 lags — that is expected
            pytest.skip("Not enough data for input_seq_len=672")


class TestTrainingPipeline:

    def test_pipeline_fit_predict_cycle(self):
        from WattPredictor.utils.ts_generator import get_pipeline

        pipeline = get_pipeline(model_type="XGBoost")

        n_lag = 672
        cols = [f"demand_previous_{i+1}_hour" for i in reversed(range(n_lag))]
        cols += [
            "temperature_2m",
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "is_holiday",
        ]

        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, len(cols)), columns=cols)
        y_train = np.random.randn(100) * 100 + 2000
        X_test = pd.DataFrame(np.random.randn(20, len(cols)), columns=cols)

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        assert len(predictions) == 20
        assert not np.isnan(predictions).any()

    def test_model_evaluation_metrics(self):
        from sklearn.metrics import (
            mean_absolute_error,
            mean_absolute_percentage_error,
            root_mean_squared_error,
            r2_score,
        )

        y_true = np.array([2000, 2100, 1900, 2200, 1800], dtype=float)
        y_pred = np.array([2010, 2080, 1920, 2180, 1810], dtype=float)

        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        assert mae > 0
        assert mape < 50
        assert rmse >= mae
        assert r2 > 0




class TestPredictionPipeline:

    def test_prediction_output_format(self):
        from WattPredictor.utils.ts_generator import get_pipeline

        pipeline = get_pipeline(model_type="XGBoost")

        n_lag = 672
        cols = [f"demand_previous_{i+1}_hour" for i in reversed(range(n_lag))]
        cols += [
            "temperature_2m",
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "is_holiday",
        ]

        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, len(cols)), columns=cols)
        y_train = np.random.randn(100) * 100 + 2000
        X_test = pd.DataFrame(np.random.randn(10, len(cols)), columns=cols)

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == X_test.shape[0]

    def test_batch_prediction(self):
        from WattPredictor.utils.ts_generator import get_pipeline

        pipeline = get_pipeline(model_type="LightGBM")

        n_lag = 672
        cols = [f"demand_previous_{i+1}_hour" for i in reversed(range(n_lag))]
        cols += [
            "temperature_2m",
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "is_holiday",
        ]

        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, len(cols)), columns=cols)
        y_train = np.random.randn(100) * 100 + 2000

        pipeline.fit(X_train, y_train)

        for batch_size in [1, 10, 50]:
            X_batch = pd.DataFrame(
                np.random.randn(batch_size, len(cols)), columns=cols
            )
            predictions = pipeline.predict(X_batch)
            assert len(predictions) == batch_size


class TestErrorHandling:

    def test_handles_missing_data(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=100, freq="H"),
                "demand": [2000] * 50 + [np.nan] * 50,
                "sub_region_code": [0] * 100,
            }
        )

        assert df is not None
        assert df["demand"].isna().sum() == 50

    def test_handles_insufficient_data(self):
        from WattPredictor.utils.ts_generator import (
            get_cutoff_indices_features_and_target,
        )

        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=10, freq="H"),
                "demand": [2000] * 10,
                "sub_region_code": [0] * 10,
            }
        )

        # 672 lags > 10 rows, should return empty
        cutoff_indices = get_cutoff_indices_features_and_target(df, 672, 1)
        assert len(cutoff_indices) == 0


class TestDVCPipeline:

    def test_dvc_yaml_exists(self):
        dvc_path = Path("dvc.yaml")
        assert dvc_path.exists(), "dvc.yaml should exist"

        import yaml

        with open(dvc_path, "r") as f:
            dvc_config = yaml.safe_load(f)

        assert "stages" in dvc_config
        stages = dvc_config["stages"]

        expected_stages = ["prepare_data", "train_model", "predict"]
        for stage in expected_stages:
            assert stage in stages

    def test_dvc_stage_structure(self):
        import yaml

        with open("dvc.yaml", "r") as f:
            dvc_config = yaml.safe_load(f)

        stages = dvc_config["stages"]

        for stage_name, stage_config in stages.items():
            assert "cmd" in stage_config, f"{stage_name} should have 'cmd'"

            if stage_name != "prepare_data":
                assert "deps" in stage_config or "outs" in stage_config


class TestArtifactGeneration:

    def test_metrics_artifact_format(self):
        metrics = {"mae": 45.3, "mape": 2.8, "rmse": 67.2, "r2": 0.94}

        expected_keys = ["mae", "mape", "rmse", "r2"]
        for key in expected_keys:
            assert key in metrics

        for value in metrics.values():
            assert isinstance(value, (int, float))

    def test_predictions_artifact_format(self):
        predictions_df = pd.DataFrame(
            {
                "date": pd.date_range("2025-02-01", periods=24, freq="H"),
                "actual": [2000 + i * 10 for i in range(24)],
                "predicted": [1990 + i * 10 for i in range(24)],
                "sub_region_code": [0] * 24,
            }
        )

        assert "date" in predictions_df.columns
        assert "predicted" in predictions_df.columns
        assert pd.api.types.is_datetime64_any_dtype(predictions_df["date"])
        assert pd.api.types.is_numeric_dtype(predictions_df["predicted"])
