import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from WattPredictor.entity.config_entity import EvaluationConfig
from WattPredictor.utils.ts_generator import features_and_target, average_demand_last_4_weeks
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
from WattPredictor.utils.helpers import create_directories, save_json
from WattPredictor.utils.exception import CustomException
from WattPredictor.utils.logging import logger
from WattPredictor.utils.business_metrics import BusinessMetrics

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def evaluate(self):
        logger.info("Starting model evaluation process")
        # Load preprocessed data directly
        preprocessed_path = Path("artifacts/engineering/preprocessed.csv")
        if not preprocessed_path.exists():
            raise CustomException(f"Preprocessed data not found: {preprocessed_path}", None)
        df = pd.read_csv(preprocessed_path)

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

        # Load model from local artifacts
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise CustomException(f"Model not found: {model_path}", None)
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

        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "mape": float(mape),
            "rmse": float(rmse),
            "r2_score": float(r2),
            "n_samples": len(test_y),
            "timestamp": datetime.now().isoformat()
        }

        # Calculate business impact
        logger.info("Calculating business impact metrics")
        business_calculator = BusinessMetrics(
            avg_demand_mw=2500,
            electricity_price_per_mwh=65,
            reserve_margin_percent=15,
            peak_capacity_cost_per_mw_year=120000
        )
        
        business_report = business_calculator.generate_report(
            rmse=rmse,
            mae=mae,
            mape=mape,
            output_path=Path(self.config.metrics_path).parent / "business_impact.json"
        )
        
        # Add business summary to metrics
        metrics["business_impact"] = {
            "annual_savings_usd": business_report["detailed_results"]["cost_savings"]["total_annual_savings_usd"],
            "roi_payback_years": business_report["detailed_results"]["roi"]["roi_payback_years"],
            "forecast_improvement_percent": business_report["detailed_results"]["forecast_improvement"]["error_reduction_percent"],
            "capacity_freed_mw": business_report["detailed_results"]["cost_savings"]["reserve_capacity_savings_mw"]
        }
        
        logger.info(f"Annual Savings: ${metrics['business_impact']['annual_savings_usd']:,.0f}")
        logger.info(f"ROI Payback: {metrics['business_impact']['roi_payback_years']:.1f} years")
        logger.info(f"Forecast Improvement: {metrics['business_impact']['forecast_improvement_percent']:.1f}%")

        create_directories([Path(self.config.metrics_path).parent])
        save_json(self.config.metrics_path, metrics)
        
        logger.info(f"Validation RMSE: {rmse:.2f} MW | MAE: {mae:.2f} MW | MAPE: {mape:.2f}%")
        logger.info(f"Metrics saved: {self.config.metrics_path}")
        
        return metrics