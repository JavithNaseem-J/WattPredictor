import os
import json
import joblib
import mlflow
import dagshub
import pandas as pd
import numpy as np
from pathlib import Path
from WattPredictor import logger
from WattPredictor.entity.config_entity import ModelEvaluationConfig
from WattPredictor.utils.helpers import create_directories, save_json, load_json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Electricity Demand Prediction")
        logger.info("MLflow tracking setup complete.")

    def evaluate(self):
            # Load model and test data
            model = joblib.load(self.config.model_path)
            x_test = np.load(self.config.x_transform, allow_pickle=True)
            y_test = np.load(self.config.y_transform, allow_pickle=True).squeeze()

            # Predict
            preds = model.predict(x_test)

            # Metrics
            metrics = {
                "mse": mean_squared_error(y_test, preds),
                "mae": mean_absolute_error(y_test, preds),
                "rmse": np.sqrt(mean_squared_error(y_test, preds)),
                "mape": np.mean(np.abs((y_test - preds) / y_test)) * 100 if np.any(y_test != 0) else np.inf,
                "r2_score": r2_score(y_test, preds),
                "adjusted_r2": 1 - (1 - r2_score(y_test, preds)) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)
            }

            create_directories([Path(self.config.metrics_path).parent])
            save_json(Path(self.config.metrics_path), metrics)
            logger.info(f"Model evaluation metrics: {metrics}")
            # Log to MLflow
            with mlflow.start_run(run_name="Model Evaluation"):
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
                mlflow.set_tag("stage", "evaluation")
                mlflow.log_artifact(self.config.metrics_path)
                mlflow.log_artifact(self.config.model_path)

            logger.info("✅ Model evaluation complete. Metrics logged.")
            return metrics
