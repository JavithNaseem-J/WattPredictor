import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from WattPredictor.utils.exception import CustomException
from WattPredictor.utils.feature import feature_store_instance
from WattPredictor.utils.helpers import create_directories
from WattPredictor.entity.config_entity import PredictionConfig
from WattPredictor.utils.ts_generator import average_demand_last_4_weeks
from WattPredictor.utils.logging import logger

class Predictor:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.feature_store = feature_store_instance()
        self.model = None
        
        # First try to load local model
        local_model_path = "artifacts/trainer/model.joblib"
        if os.path.exists(local_model_path):
            try:
                self.model = joblib.load(local_model_path)
                logger.info(f"Loaded local model from {local_model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load local model: {str(e)}")
        
        # Fall back to Hopsworks model registry
        try:
            model_registry = self.feature_store.project.get_model_registry()
            model_names = ["wattpredictor_xgboost", "wattpredictor_lightgbm"]
            best_model = None
            best_rmse = float("inf")
            best_model_name = None
            best_model_version = None

            all_models = []
            for model_name in model_names:
                models = model_registry.get_models(model_name)
                if models:
                    all_models.extend([(model, model_name) for model in models])

            if not all_models:
                raise CustomException("No models found with names 'wattpredictor_xgboost' or 'wattpredictor_lightgbm'", None)

            for model, model_name in all_models:
                try:
                    rmse = model.metrics.get("rmse", float("inf")) if hasattr(model, 'metrics') else float("inf")
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model
                        best_model_name = model_name
                        best_model_version = model.version
                except AttributeError:
                    logger.warning(f"Model {model_name} v{model.version} has no 'metrics' attribute")

            if best_model is None:
                # Fallback to the latest version
                best_model, best_model_name = max(all_models, key=lambda x: x[0].version)
                best_model_version = best_model.version
                logger.warning(f"No metrics available, using latest model: {best_model_name} v{best_model_version}")

            self.model = self.feature_store.load_model(
                model_name=best_model_name,
                model_version=best_model_version,
                model_filename='model.joblib'
            )
            logger.info(f"Loaded model {best_model_name} v{best_model_version} with RMSE {best_rmse if best_rmse != float('inf') else 'unknown'}")
        except Exception as e:
            logger.error(f"Failed to load model from Hopsworks: {str(e)}")
            raise CustomException("Could not load model from local path or Hopsworks", e)

    def _load_batch_features(self, current_date):
        logger.info(f"Fetching features up to {current_date}")
        feature_view = self.feature_store.feature_store.get_feature_view(
            name=self.config.feature_view_name,
            version=self.config.feature_view_version
        )
        fetch_data_to = current_date
        fetch_data_from = current_date - timedelta(days=29)
        try:
            ts_data = feature_view.get_batch_data(
                start_time=fetch_data_from,
                end_time=fetch_data_to
            )
        except Exception as e:
            logger.warning(f"Failed to fetch data up to {fetch_data_to}: {str(e)}. Falling back to previous hour.")
            fetch_data_to = current_date - timedelta(hours=1)
            ts_data = feature_view.get_batch_data(
                start_time=fetch_data_from,
                end_time=fetch_data_to
            )

        ts_data = ts_data.groupby('sub_region_code').tail(self.config.n_features)
        ts_data.sort_values(by=['sub_region_code', 'date'], inplace=True)

        location_ids = ts_data['sub_region_code'].unique()
        x = np.ndarray((len(location_ids), self.config.n_features), dtype=np.float32)
        additional_features = {
            'temperature_2m': [], 'hour': [], 'day_of_week': [], 'month': [], 
            'is_weekend': [], 'is_holiday': []
        }

        for i, loc in enumerate(location_ids):
            sub_data = ts_data[ts_data['sub_region_code'] == loc]
            demand_series = sub_data['demand'].values[-self.config.n_features:]
            if len(demand_series) < self.config.n_features:
                demand_series = np.pad(demand_series, 
                                     (self.config.n_features - len(demand_series), 0), 
                                     'constant', constant_values=0)
            x[i, :] = demand_series
            for col in additional_features:
                additional_features[col].append(sub_data[col].iloc[-1] if col in sub_data else 0)

        features = pd.DataFrame(
            x, columns=[f'demand_previous_{i+1}_hour' for i in reversed(range(self.config.n_features))]
        )
        for col in additional_features:
            features[col] = additional_features[col]
        features['date'] = current_date
        features['sub_region_code'] = location_ids
        features = average_demand_last_4_weeks(features)
        logger.info(f"Features prepared for {len(location_ids)} sub-regions")
        return features

    def save_predictions_to_store(self, predictions: pd.DataFrame):
        if predictions.empty:
            logger.warning("No predictions to save to feature store")
            return
        self.feature_store.create_feature_group(
            name='elec_wx_predictions',
            df=predictions,
            primary_key=["sub_region_code"],
            event_time="date",
            description="Predicted electricity demand",
            online_enabled=True
        )
        logger.info("Predictions saved to feature store")

    def predict(self, save_to_store: bool = False) -> pd.DataFrame:
        current_date = datetime.now().replace(minute=0, second=0, microsecond=0)
        features = self._load_batch_features(current_date)
        feature_input = features.drop(columns=['date', 'sub_region_code'], errors='ignore')
        predictions = self.model.predict(feature_input)
        predictions_df = pd.DataFrame({
            'sub_region_code': features['sub_region_code'],
            'predicted_demand': predictions.round(0),
            'date': current_date
        })
        if save_to_store:
            self.save_predictions_to_store(predictions_df)
        create_directories([self.config.predictions_df.parent])
        predictions_df.to_csv(self.config.predictions_df, index=False)
        logger.info(f"Predictions generated for {current_date} with {len(predictions_df)} records")
        return predictions_df