import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from WattPredictor.config.config import WattPredictorConfig
from WattPredictor.components.training.trainer import Trainer
from WattPredictor.components.training.evaluator import Evaluation
from WattPredictor.utils.ts_generator import get_pipeline, features_and_target, get_cutoff_indices_features_and_target, average_demand_last_4_weeks


@pytest.fixture
def predictor_config(tmp_path):
    config = WattPredictorConfig(project_root=tmp_path)
    return config


class TestTrainer:

    def test_trainer_initialization(self, predictor_config):
        trainer = Trainer(config=predictor_config)
        assert trainer is not None
        assert hasattr(trainer, "param_grids")
        assert "XGBoost" in trainer.param_grids
        assert "LightGBM" in trainer.param_grids

    def test_xgboost_params_structure(self, predictor_config):
        trainer = Trainer(config=predictor_config)
        params = trainer.param_grids["XGBoost"]

        assert "model__n_estimators" in params
        assert "model__max_depth" in params
        assert "model__learning_rate" in params

        for depth in params["model__max_depth"]:
            assert 3 <= depth <= 10
        for lr in params["model__learning_rate"]:
            assert 0.001 <= lr <= 0.3

    def test_lightgbm_params_structure(self, predictor_config):
        trainer = Trainer(config=predictor_config)
        params = trainer.param_grids["LightGBM"]

        assert "model__num_leaves" in params
        assert "model__learning_rate" in params
        assert "model__n_estimators" in params

    def test_xgboost_pipeline_fit_predict(self):
        pipeline = get_pipeline(model_type="XGBoost")

        n_lag = 672
        cols = [f"demand_previous_{i+1}_hour" for i in reversed(range(n_lag))]
        cols += ["temperature_2m", "hour", "day_of_week", "month", "is_weekend", "is_holiday"]
        n_cols = len(cols)

        np.random.seed(42)
        n_train = 100
        X_train = pd.DataFrame(np.random.randn(n_train, n_cols), columns=cols)
        y_train = np.random.randn(n_train) * 100 + 2000

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_train[:10])

        assert len(predictions) == 10
        assert not np.isnan(predictions).any()

    def test_lightgbm_pipeline_fit_predict(self):
        pipeline = get_pipeline(model_type="LightGBM")

        n_lag = 672
        cols = [f"demand_previous_{i+1}_hour" for i in reversed(range(n_lag))]
        cols += ["temperature_2m", "hour", "day_of_week", "month", "is_weekend", "is_holiday"]

        np.random.seed(42)
        n_train = 100
        X_train = pd.DataFrame(np.random.randn(n_train, len(cols)), columns=cols)
        y_train = np.random.randn(n_train) * 100 + 2000

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_train[:10])

        assert len(predictions) == 10
        assert not np.isnan(predictions).any()

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="model_type must be"):
            get_pipeline(model_type="RandomForest")

    def test_save_and_load_model(self, tmp_path):
        pipeline = get_pipeline(model_type="XGBoost")

        n_lag = 672
        cols = [f"demand_previous_{i+1}_hour" for i in reversed(range(n_lag))]
        cols += ["temperature_2m", "hour", "day_of_week", "month", "is_weekend", "is_holiday"]

        np.random.seed(42)
        n_train = 100
        X_train = pd.DataFrame(np.random.randn(n_train, len(cols)), columns=cols)
        y_train = np.random.randn(n_train) * 100 + 2000

        pipeline.fit(X_train, y_train)
        pred_before = pipeline.predict(X_train[:5])

        model_path = tmp_path / "test_model.joblib"
        joblib.dump(pipeline, model_path)

        loaded = joblib.load(model_path)
        pred_after = loaded.predict(X_train[:5])

        np.testing.assert_array_almost_equal(pred_before, pred_after)


class TestEvaluation:

    def test_evaluation_initialization(self, predictor_config):
        evaluator = Evaluation(config=predictor_config)
        assert evaluator is not None
        assert evaluator.config.input_seq_len == 672

    def test_sklearn_metrics_calculation(self):
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            mean_absolute_percentage_error,
            root_mean_squared_error,
            r2_score,
        )

        y_true = np.array([100, 200, 300, 400, 500], dtype=float)
        y_pred = np.array([110, 190, 310, 390, 510], dtype=float)

        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        assert mae == 10.0
        assert rmse >= mae
        assert 0 < mape < 100
        assert 0 < r2 <= 1

    def test_r2_perfect_prediction(self):
        from sklearn.metrics import r2_score

        y_true = np.array([100, 200, 300, 400, 500], dtype=float)
        y_pred = y_true.copy()

        r2 = r2_score(y_true, y_pred)
        assert abs(r2 - 1.0) < 0.001

    def test_mape_calculation(self):
        from sklearn.metrics import mean_absolute_percentage_error

        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])

        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        assert 6.0 <= mape <= 6.2

    def test_save_metrics_json(self, tmp_path):
        from WattPredictor.utils.helpers import save_json, load_json

        metrics = {"mae": 50.0, "mape": 2.5, "rmse": 75.0, "r2": 0.95}
        save_path = tmp_path / "metrics.json"

        save_json(str(save_path), metrics)
        assert save_path.exists()

        loaded = load_json(str(save_path))
        assert loaded == metrics


class TestModelPersistence:

    def test_save_and_load_pipeline(self, tmp_path):
        pipeline = get_pipeline(model_type="XGBoost")

        n_lag = 672
        cols = [f"demand_previous_{i+1}_hour" for i in reversed(range(n_lag))]
        cols += ["temperature_2m", "hour", "day_of_week", "month", "is_weekend", "is_holiday"]

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, len(cols)), columns=cols)
        y = np.random.randn(100) * 100 + 2000

        pipeline.fit(X, y)
        pred_before = pipeline.predict(X[:5])

        model_path = tmp_path / "model.joblib"
        joblib.dump(pipeline, model_path)
        loaded = joblib.load(model_path)
        pred_after = loaded.predict(X[:5])

        np.testing.assert_array_almost_equal(pred_before, pred_after)

    def test_model_metadata_package(self, tmp_path):
        pipeline = get_pipeline(model_type="LightGBM")

        n_lag = 672
        cols = [f"demand_previous_{i+1}_hour" for i in reversed(range(n_lag))]
        cols += ["temperature_2m", "hour", "day_of_week", "month", "is_weekend", "is_holiday"]

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, len(cols)), columns=cols)
        y = np.random.randn(100) * 100 + 2000

        pipeline.fit(X, y)

        package = {
            "model": pipeline,
            "model_type": "LightGBM",
            "features": cols,
            "metrics": {"mae": 50.0, "mape": 2.5},
        }

        pkg_path = tmp_path / "model_package.joblib"
        joblib.dump(package, pkg_path)

        loaded = joblib.load(pkg_path)
        assert loaded["model_type"] == "LightGBM"
        assert loaded["metrics"]["mape"] == 2.5
        assert loaded["model"] is not None
