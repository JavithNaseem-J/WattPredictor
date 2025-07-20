import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from WattPredictor.entity.config_entity import EvaluationConfig
from WattPredictor.utils.feature import feature_store_instance
from WattPredictor.utils.ts_generator import features_and_target, average_demand_last_4_weeks
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
from WattPredictor.utils.helpers import create_directories, save_json
from WattPredictor.utils.exception import CustomException
from WattPredictor.utils.logging import logger

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.feature_store = feature_store_instance()

    def evaluate(self):
        df, _ = self.feature_store.get_training_data("elec_wx_features_view")
        df = df[['date', 'demand', 'sub_region_code', 'temperature_2m', 
                 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']]
        df.sort_values("date", inplace=True)
        if df.empty:
            raise CustomException("Loaded DataFrame is empty", None)

        cutoff_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        train_df, test_df = df[df['date'] < cutoff_date], df[df['date'] >= cutoff_date]
        if test_df.empty:
            raise CustomException("Test DataFrame is empty after applying cutoff_date", None)

        test_x, test_y = features_and_target(test_df, input_seq_len=self.config.input_seq_len, step_size=self.config.step_size)
        test_x.drop(columns=["date"], errors="ignore", inplace=True)

        non_numeric_cols = test_x.select_dtypes(exclude=['int64', 'float64', 'bool']).columns
        if not non_numeric_cols.empty:
            raise CustomException(f"Non-numeric columns found in test_x: {non_numeric_cols}", None)

        model_registry = self.feature_store.project.get_model_registry()
        model_name = "wattpredictor_xgboost"
        models = model_registry.get_models(model_name)
        if not models:
            model_name = "wattpredictor_lightgbm"
            models = model_registry.get_models(model_name)
            if not models:
                raise CustomException(f"No models found with names 'wattpredictor_xgboost' or 'wattpredictor_lightgbm'", None)
        
        latest_model = max(models, key=lambda m: m.version)
        model_dir = latest_model.download()
        model_path = Path(model_dir) / "model.joblib"
        pipeline = joblib.load(model_path)

        test_x_transformed = test_x.copy()
        test_x_transformed = average_demand_last_4_weeks(test_x_transformed)
        preds = pipeline.predict(test_x_transformed)

        mse = mean_squared_error(test_y, preds)
        mae = mean_absolute_error(test_y, preds)
        mape = mean_absolute_percentage_error(test_y, preds) * 100
        rmse = root_mean_squared_error(test_y, preds)
        r2 = r2_score(test_y, preds)
        adjusted_r2 = 1 - (1 - r2) * (len(test_y) - 1) / (len(test_y) - test_x_transformed.shape[1] - 1)

        metrics = {
            "mse": mse,
            "mae": mae,
            "mape": mape,
            "rmse": rmse,
            "r2_score": r2,
            "adjusted_r2": adjusted_r2
        }

        create_directories([Path(self.config.metrics_path).parent])
        save_json(self.config.metrics_path, metrics)
        create_directories([Path(self.config.img_path).parent])
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_y[:100], label="Actual", color="blue")
        ax.plot(preds[:100], label="Predicted", color="red")
        ax.set_title("Predicted vs Actual (First 100 Points)")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Electricity Demand")
        ax.legend()
        fig.savefig(self.config.img_path)
        plt.close()

        self.feature_store.upload_file_safely(self.config.metrics_path, "eval/metrics.json")
        self.feature_store.upload_file_safely(self.config.img_path, "eval/pred_vs_actual.png")
        logger.info("Evaluation results uploaded to Hopsworks")
        return metrics