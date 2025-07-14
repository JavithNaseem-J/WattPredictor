import os
import sys
import json
import joblib
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from WattPredictor.config.model_config import ModelEvaluationConfig
from WattPredictor.config.feature_config import FeatureStoreConfig
from WattPredictor.components.feature_store import FeatureStore
from WattPredictor.utils.ts_generator import features_and_target
from sklearn.metrics import mean_squared_error, mean_absolute_error,root_mean_squared_error, r2_score
from WattPredictor.utils.helpers import create_directories, save_json
from WattPredictor.utils.exception import CustomException
from WattPredictor import logger

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, feature_store_config):
        self.config = config
        self.feature_store = FeatureStore(feature_store_config)

    def evaluate(self):
        try:
            df, _ = self.feature_store.load_latest_training_dataset("elec_wx_features_view")
            df = df[['date', 'demand', 'sub_region_code', 'temperature_2m']]
            df.sort_values("date", inplace=True)
            _, test_df = df[df['date'] >= self.config.cutoff_date], df[df['date'] < self.config.cutoff_date]

            test_x, test_y = features_and_target(test_df, input_seq_len=self.config.input_seq_len, step_size=self.config.step_size)
            test_x.drop(columns=["date"], errors="ignore", inplace=True)

            # Load model from Hopsworks
            model_registry = self.feature_store.project.get_model_registry()
            model = model_registry.get_model("wattpredictor_xgboost", version=1)
            model_dir = model.download()
            model_path = os.path.join(model_dir, "model.joblib")
            model_instance = joblib.load(model_path)


            preds = model_instance.predict(test_x)

            metrics = {
                "mse": mean_squared_error(test_y, preds),
                "mae": mean_absolute_error(test_y, preds),
                "rmse": root_mean_squared_error(test_y, preds),
                "mape": np.mean(np.abs((test_y - preds) / test_y)) * 100 if np.any(test_y != 0) else np.inf,
                "r2_score": r2_score(test_y, preds),
                "adjusted_r2": 1 - (1 - r2_score(test_y, preds)) * (len(test_y) - 1) / (len(test_y) - test_x.shape[1] - 1)
            }

            create_directories([os.path.dirname(self.config.metrics_path)])
            save_json(self.config.metrics_path, metrics)

            logger.info(f"Evaluation complete. Metrics: {metrics}")
            return metrics

        except Exception as e:
            raise CustomException(e, sys)