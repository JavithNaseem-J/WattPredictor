import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from WattPredictor.utils.logging import logger
from WattPredictor.utils.exception import CustomException
from WattPredictor.utils.feature import feature_store_instance
from WattPredictor.entity.config_entity import PredictionConfig
from WattPredictor.config.inference_config import InferenceConfigurationManager


class Predictor:
    def __init__(self,config: PredictionConfig):
        self.config = config
        self.feature_store = feature_store_instance()

        self.model = self.feature_store.load_model(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            model_filename='model.joblib'
        )

    def _load_batch_features(self, current_date: datetime) -> pd.DataFrame:
        try:
            feature_view = self.feature_store.feature_store.get_feature_view(
                name=self.config.feature_view_name,
                version=self.config.feature_view_version
            )

            fetch_data_to = current_date - timedelta(hours=1)
            fetch_data_from = current_date - timedelta(days=28)

            ts_data = feature_view.get_batch_data(
                start_time=fetch_data_from,
                end_time=fetch_data_to
            )

            ts_data = ts_data.groupby('sub_region_code').tail(self.config.n_features)
            ts_data.sort_values(by=['sub_region_code', 'date'], inplace=True)

            location_ids = ts_data['sub_region_code'].unique()
            x = np.ndarray((len(location_ids), self.config.n_features), dtype=np.float32)
            temperature_values = []

            for i, loc in enumerate(location_ids):
                sub_data = ts_data[ts_data['sub_region_code'] == loc]
                demand_series = sub_data['demand'].values[-self.config.n_features:]

                if len(demand_series) < self.config.n_features:
                    logger.warning(f"Padded {loc}: {len(demand_series)} available, padding to {self.config.n_features}.")
                    demand_series = np.pad(demand_series, (self.config.n_features - len(demand_series), 0), 'constant', constant_values=0)

                x[i, :] = demand_series
                temperature_values.append(sub_data['temperature_2m'].iloc[-1])

            features = pd.DataFrame(
                x, columns=[f'demand_previous_{i+1}_hour' for i in reversed(range(self.config.n_features))]
            )
            features['temperature_2m'] = temperature_values
            features['date'] = current_date
            features['sub_region_code'] = location_ids

            logger.info(f"Features generated for {len(location_ids)} regions.")
            return features

        except Exception as e:
            logger.error("Failed to load batch features.")
            raise CustomException(e, sys)

    def predict(self, current_date: datetime, save_to_store: bool = False) -> pd.DataFrame:
        try:
            features = self._load_batch_features(current_date)
            feature_input = features.drop(columns=['date'], errors='ignore')
            predictions = self.model.predict(feature_input)

            results = pd.DataFrame({
                'sub_region_code': features['sub_region_code'],
                'predicted_demand': predictions.round(0),
                'date': features['date']
            })

            logger.info("Predictions generated successfully.")

            if save_to_store:
                self.save_predictions_to_store(results)

            return results

        except Exception as e:
            logger.error("Prediction process failed.")
            raise CustomException(e, sys)

    def save_predictions_to_store(self, predictions: pd.DataFrame):
        try:
            if predictions.empty:
                logger.warning("No predictions to save.")
                return

            self.feature_store.create_feature_group(
                name='elec_wx_predictions',
                df=predictions,
                primary_key=["sub_region_code"],
                event_time="date",
                description="Predicted electricity demand",
                online_enabled=True
            )

            logger.info("Predictions saved to feature store.")

        except Exception as e:
            logger.error("Failed to save predictions to feature store.")
            raise CustomException(e, sys)