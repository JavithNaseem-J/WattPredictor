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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

from WattPredictor.utils.helpers import create_directories, save_json
from WattPredictor.utils.exception import CustomException
from WattPredictor import logger


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, feature_store_config):
        self.config = config
        self.feature_store_config = feature_store_config
        self.feature_store = FeatureStore(feature_store_config)

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Electricity Demand Prediction")
        logger.info("MLflow tracking setup complete.")

    def download_inputs(self):
        try:

            self.feature_store.dataset_api.download("Resources/wattpredictor_artifacts/model.joblib/model.joblib", overwrite=True)
            self.feature_store.dataset_api.download("Resources/wattpredictor_artifacts/test_x.parquet/test_x.parquet", overwrite=True)
            self.feature_store.dataset_api.download("Resources/wattpredictor_artifacts/test_y.parquet/test_y.parquet", overwrite=True)

            test_x = pd.read_parquet(self.config.x_transform)
            test_y = pd.read_parquet(self.config.y_transform)
            
            test_x = test_x.values
            test_y = test_y.squeeze().values 
            model = joblib.load(self.config.model_path)

            logger.info(f'shape of train_x:{test_x.shape}, train_y:{test_y.shape}')

            return test_x,test_y, model

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate(self):
        try:
            test_x,test_y, model = self.download_inputs()


            # Predict
            preds = model.predict(test_x)

            # Metrics
            metrics = {
                "mse": mean_squared_error(test_y, preds),
                "mae": mean_absolute_error(test_y, preds),
                "rmse": np.sqrt(mean_squared_error(test_y, preds)),
                "mape": np.mean(np.abs((test_y - preds) / test_y)) * 100 if np.any(test_y != 0) else np.inf,
                "r2_score": r2_score(test_y, preds),
                "adjusted_r2": 1 - (1 - r2_score(test_y, preds)) * (len(test_y) - 1) / (len(test_y) - test_x.shape[1] - 1)
            }

            create_directories([Path(self.config.metrics_path).parent])
            save_json(Path(self.config.metrics_path), metrics)

            logger.info(f"Evaluation Metrics: {metrics}")

            # Log to MLflow
            with mlflow.start_run(run_name="Model Evaluation"):
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
                mlflow.set_tag("stage", "evaluation")
                mlflow.log_artifact(self.config.metrics_path)
                mlflow.log_artifact(self.config.model_path)

            logger.info("Model evaluation complete and metrics logged.")
            return metrics

        except Exception as e:
            raise CustomException(e, sys)