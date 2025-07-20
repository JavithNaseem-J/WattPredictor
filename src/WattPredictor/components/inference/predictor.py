import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from WattPredictor.utils.exception import CustomException
from WattPredictor.utils.feature import feature_store_instance
from WattPredictor.utils.helpers import create_directories
from WattPredictor.entity.config_entity import PredictionConfig
from WattPredictor.config.inference_config import InferenceConfigurationManager
from WattPredictor.utils.ts_generator import average_demand_last_4_weeks
from WattPredictor.utils.logging import logger

class Predictor:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.feature_store = feature_store_instance()
        self.model = self.feature_store.load_model(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            model_filename='model.joblib'
        )

    def _load_batch_features(self, current_date):
        feature_view = self.feature_store.feature_store.get_feature_view(
            name=self.config.feature_view_name,
            version=self.config.feature_view_version
        )
        fetch_data_to = datetime.now() - timedelta(hours=1)
        fetch_data_from = datetime.now() - timedelta(days=29)
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
                additional_features[col].append(sub_data[col].iloc[-1])

        features = pd.DataFrame(
            x, columns=[f'demand_previous_{i+1}_hour' for i in reversed(range(self.config.n_features))]
        )
        for col in additional_features:
            features[col] = additional_features[col]
        features['date'] = (datetime.now() - timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0)
        features['sub_region_code'] = location_ids
        features = average_demand_last_4_weeks(features)
        return features

    def save_predictions_to_store(self, predictions: pd.DataFrame):
        if predictions.empty:
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
        features = self._load_batch_features(datetime.now())
        feature_input = features.drop(columns=['date', 'sub_region_code'], errors='ignore')
        predictions = self.model.predict(feature_input)
        predictions_df = pd.DataFrame({
            'sub_region_code': features['sub_region_code'],
            'predicted_demand': predictions.round(0),
            'date': (datetime.now() - timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0)
        })
        if save_to_store:
            self.save_predictions_to_store(predictions_df)
        create_directories([self.config.predictions_df.parent])
        predictions_df.to_csv(self.config.predictions_df, index=False)
        logger.info("Predictions generated successfully")
        return predictions_df