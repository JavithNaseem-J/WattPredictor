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
        logger.info("Starting model evaluation process")
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
        model_names = ["wattpredictor_xgboost", "wattpredictor_lightgbm"]
        all_models = []
        
        for model_name in model_names:
            models = model_registry.get_models(model_name)
            if models:
                all_models.extend([(model, model_name) for model in models])
        
        if not all_models:
            raise CustomException("No models found with names 'wattpredictor_xgboost' or 'wattpredictor_lightgbm'", None)
        
        selected_model = None
        selected_model_name = None
        best_rmse = float("inf")

        for model, model_name in all_models:
            try:
                rmse = model.metrics.get("rmse", float("inf")) if hasattr(model, 'metrics') else float("inf")
                if rmse < best_rmse:
                    best_rmse = rmse
                    selected_model = model
                    selected_model_name ==model_name
            except AttributeError:
                logger.warning(f"Model {model_name} v{model.version} has no 'metrics' attribute")
        
        if selected_model is None:
            # Fallback to the latest version
            selected_model, selected_model_name = max(all_models, key=lambda x: x[0].version)
            logger.warning(f"No metrics available, using latest model: {selected_model_name} v{selected_model.version}")

        model_dir = selected_model.download()
        model_path = Path(model_dir) / "model.joblib"
        pipeline = joblib.load(model_path)
        logger.info(f"Loaded model {selected_model_name} v{selected_model.version} for evaluation")

        test_x_transformed = test_x.copy()
        test_x_transformed = average_demand_last_4_weeks(test_x_transformed)
        preds = pipeline.predict(test_x_transformed)

        mse = mean_squared_error(test_y, preds)
        mae = mean_absolute_error(test_y, preds)
        mape = mean_absolute_percentage_error(test_y, preds) * 100
        rmse = root_mean_squared_error(test_y, preds)
        r2 = r2_score(test_y, preds)

        metrics = {
            "model_name": selected_model_name,
            "model_version": selected_model.version,
            "mse": mse,
            "mae": mae,
            "mape": mape,
            "rmse": rmse,
            "r2_score": r2,
        }

        create_directories([Path(self.config.metrics_path).parent])
        save_json(self.config.metrics_path, metrics)
        create_directories([Path(self.config.img_path).parent])
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_y[:250], label="Actual", color="blue")
        ax.plot(preds[:250], label="Predicted", color="red")
        ax.set_title(f"Predicted vs Actual (First 100 Points) - {selected_model_name} v{selected_model.version}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Electricity Demand (MWh)")
        ax.legend()
        fig.savefig(self.config.img_path)
        plt.close()

        self.feature_store.upload_file_safely(self.config.metrics_path, f"eval/metrics_{selected_model_name}_v{selected_model.version}.json")
        self.feature_store.upload_file_safely(self.config.img_path, f"eval/pred_vs_actual_{selected_model_name}_v{selected_model.version}.png")
        logger.info(f"Evaluation completed for {selected_model_name} v{selected_model.version} with RMSE {rmse:.4f}")
        return metrics