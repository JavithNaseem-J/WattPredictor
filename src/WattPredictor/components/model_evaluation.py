import os
import sys
import json
import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
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
            
            train_df, test_df = df[df['date'] < self.config.cutoff_date], df[df['date'] >= self.config.cutoff_date]

            test_x, test_y = features_and_target(test_df, input_seq_len=self.config.input_seq_len, step_size=self.config.step_size)
            test_x.drop(columns=["date"], errors="ignore", inplace=True)

            model_registry = self.feature_store.project.get_model_registry()
            model_name = "wattpredictor_lightgbm"
            
            models = model_registry.get_models(model_name)
            if not models:
                raise CustomException(f"No models found with name '{model_name}'", sys)
            
            latest_model = models[0] 
            
            
            model_dir = latest_model.download()
            model_path = os.path.join(model_dir, "model.joblib")
            model_instance = joblib.load(model_path)

            preds = model_instance.predict(test_x)

            mse = mean_squared_error(test_y, preds)
            mae = mean_absolute_error(test_y, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(test_y, preds)
            mape = np.mean(np.abs((test_y - preds) / test_y)) * 100 if np.any(test_y != 0) else np.inf
            adjusted_r2 = 1 - (1 - r2) * (len(test_y) - 1) / (len(test_y) - test_x.shape[1] - 1)

            metrics = {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "r2_score": r2,
                "adjusted_r2": adjusted_r2
            }

            create_directories([os.path.dirname(self.config.metrics_path)])
            save_json(self.config.metrics_path, metrics)
            logger.info(f"Saved evaluation metrics at {self.config.metrics_path}")

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(test_y[:100], label="Actual", color="blue")
            ax.plot(preds[:100], label="Predicted", color="red")
            ax.set_title("Predicted vs Actual (First 100 Points)")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Electricity Demand")
            ax.legend()

            create_directories([os.path.dirname(self.config.img_path)])
            fig.savefig(self.config.img_path)
            plt.close()
            logger.info(f"Saved prediction plot at {self.config.img_path}")

            self.feature_store.upload_file_safely(self.config.metrics_path, "eval/metrics.json")
            self.feature_store.upload_file_safely(self.config.img_path, "eval/pred_vs_actual.png")

            logger.info("Evaluation results uploaded to Hopsworks dataset storage")

            return metrics

        except Exception as e:
            raise CustomException("Model evaluation failed", e)