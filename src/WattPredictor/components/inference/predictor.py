import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from WattPredictor.utils.exception import CustomException
from WattPredictor.utils.helpers import create_directories
from WattPredictor.entity.config_entity import PredictionConfig
from WattPredictor.utils.ts_generator import average_demand_last_4_weeks
from WattPredictor.utils.logging import logger

class Predictor:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model = None
        
        # Load model from local artifacts
        model_path = str(config.model_path)
        if not os.path.exists(model_path):
            raise CustomException(f"Model not found: {model_path}", None)
        
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

    def _load_batch_features(self, current_date):
        logger.info(f"Loading features up to {current_date}")
        preprocessed_path = "artifacts/engineering/preprocessed.csv"
        if not os.path.exists(preprocessed_path):
            raise CustomException(f"Preprocessed data not found: {preprocessed_path}", None)
        
        df = pd.read_csv(preprocessed_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter to recent 29 days
        fetch_data_from = current_date - timedelta(days=29)
        df = df[(df['date'] >= fetch_data_from) & (df['date'] <= current_date)]
        
        if df.empty:
            raise CustomException("No data available in the recent 29-day window", None)
        
        location_ids = df['sub_region_code'].unique()
        feature_cols = ['temperature_2m', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']
        
        # Build features per sub-region using last available row
        rows = []
        for loc in location_ids:
            sub_data = df[df['sub_region_code'] == loc].sort_values('date')
            if sub_data.empty:
                continue
            last_row = sub_data.iloc[-1]
            row = {'sub_region_code': loc, 'date': current_date}
            # Use demand values as lag features
            demand_vals = sub_data['demand'].values
            n_lags = min(len(demand_vals), 672)  # up to input_seq_len
            for i in range(n_lags):
                row[f'demand_previous_{n_lags - i}_hour'] = demand_vals[-(n_lags - i)]
            for col in feature_cols:
                row[col] = last_row[col] if col in sub_data.columns else 0
            rows.append(row)
        
        features = pd.DataFrame(rows)
        features = average_demand_last_4_weeks(features)
        logger.info(f"Features prepared for {len(location_ids)} sub-regions")
        return features

    def predict(self, save_to_store: bool = False) -> pd.DataFrame:
        current_date = pd.Timestamp.now(tz="UTC").replace(minute=0, second=0, microsecond=0)
        features = self._load_batch_features(current_date)
        feature_input = features.drop(columns=['date', 'sub_region_code'], errors='ignore')
        predictions = self.model.predict(feature_input)
        predictions_df = pd.DataFrame({
            'sub_region_code': features['sub_region_code'],
            'predicted_demand': predictions.round(0),
            'date': current_date
        })
        create_directories([self.config.predictions_df.parent])
        predictions_df.to_csv(self.config.predictions_df, index=False)
        logger.info(f"Predictions generated for {current_date} with {len(predictions_df)} records")
        return predictions_df