import joblib
import pandas as pd
import numpy as np
import mlflow
from datetime import timedelta
from pathlib import Path
from WattPredictor.config.config import WattPredictorConfig, get_config
from WattPredictor.utils.ts_generator import features_and_target, average_demand_last_4_weeks
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
from WattPredictor.utils.helpers import create_directories, save_json
from WattPredictor.utils.logging import logger


class Evaluation:
    def __init__(self, config: WattPredictorConfig = None):
        self.config = config or get_config()

    def evaluate(self):
        logger.info("Starting model evaluation process")
        preprocessed_path = Path(self.config.preprocessed_data_path)
        if not preprocessed_path.exists():
            raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_path}")
        df = pd.read_csv(preprocessed_path)

        df = df[['date', 'demand', 'sub_region_code', 'temperature_2m', 
                 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']]
        
        df.sort_values("date", inplace=True)

        if df.empty:
            raise ValueError("Loaded DataFrame is empty")

        df_date = pd.to_datetime(df['date'])
        max_date = df_date.max()
        cutoff_date = max_date - timedelta(days=90)
        
        train_df = df[df_date < cutoff_date]
        test_df = df[df_date >= cutoff_date]

        if test_df.empty:
            raise ValueError("Test DataFrame is empty after applying dataset cutoff_date")

        test_x, test_y = features_and_target(test_df, input_seq_len=self.config.input_seq_len, step_size=self.config.step_size)
        test_x.drop(columns=["date"], errors="ignore", inplace=True)

        non_numeric_cols = test_x.select_dtypes(exclude=['int64', 'float64', 'bool']).columns
        if not non_numeric_cols.empty:
            raise ValueError(f"Non-numeric columns found in test_x: {non_numeric_cols}")

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        pipeline = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path} for evaluation")

        test_x_transformed = test_x.copy()
        test_x_transformed = average_demand_last_4_weeks(test_x_transformed)
        preds = pipeline.predict(test_x_transformed)

        mse = mean_squared_error(test_y, preds)
        mae = mean_absolute_error(test_y, preds)
        mape = mean_absolute_percentage_error(test_y, preds) * 100
        rmse = root_mean_squared_error(test_y, preds)
        r2 = r2_score(test_y, preds)

        baseline_mape = 10.0
        error_reduction_percent = ((baseline_mape - mape) / baseline_mape) * 100

        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "mape": float(mape),
            "rmse": float(rmse),
            "r2_score": float(r2),
            "baseline_mape": baseline_mape,
            "forecast_improvement_percent": float(error_reduction_percent),
            "n_samples": len(test_y)
        }

        mlflow.set_experiment("WattPredictor")
        with mlflow.start_run(run_name="Evaluation_Metrics"):
            mlflow.log_metrics({
                "val_mse": float(mse),
                "val_mae": float(mae),
                "val_mape": float(mape),
                "val_rmse": float(rmse),
                "val_r2": float(r2),
                "forecast_improvement_percent": float(error_reduction_percent)
            })
        
        logger.info(f"Validation RMSE: {rmse:.2f} MW | MAE: {mae:.2f} MW | MAPE: {mape:.2f}% | R2: {r2:.4f}")
        logger.info(f"Forecast Error Reduction vs 10% Baseline: {error_reduction_percent:.1f}%")

        create_directories([Path(self.config.metrics_path).parent])
        save_json(self.config.metrics_path, metrics)
        logger.info(f"Metrics saved: {self.config.metrics_path}")
        
        return metrics