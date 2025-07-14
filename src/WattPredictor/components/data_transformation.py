import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from WattPredictor.config.feature_config import FeatureStoreConfig
from WattPredictor.config.data_config import DataTransformationConfig
from WattPredictor.components.feature_store import FeatureStore
from WattPredictor.utils.helpers import create_directories, save_bin
from WattPredictor.utils.exception import CustomException
from WattPredictor import logger


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, feature_store_config: FeatureStoreConfig):
        self.config = config
        self.feature_store = FeatureStore(feature_store_config)

    def check_status(self):
        try:
            with open(self.config.status_file, 'r') as f:
                status_data = json.load(f)
            return status_data.get("validation_status", False)
        except Exception as e:
            logger.warning(f"Validation status check failed: {e}")
            return False

    def basic_preprocessing(self) -> pd.DataFrame:
        try:
            fg = self.feature_store.feature_store.get_feature_group(name="elec_wx_demand", version=1)
            df = fg.read()
            le = LabelEncoder()
            df['sub_region_code'] = le.fit_transform(df['subba'])
            df.rename(columns={'subba': 'sub_region', 'value': 'demand'}, inplace=True)
            df = df[['date_str','date', 'sub_region_code', 'demand', 'temperature_2m']]

            create_directories([os.path.dirname(self.config.label_encoder)])
            save_bin(le, self.config.label_encoder)
            self.feature_store.upload_file_safely(self.config.label_encoder, "label_encoder.pkl")

            logger.info("Label encoding and preprocessing complete.")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['date'] = pd.to_datetime(df['date'], utc=True)
            
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

            holidays = calendar().holidays(start=df['date'].min(), end=df['date'].max())
            df['is_holiday'] = df['date'].isin(holidays).astype(int)

            
            self.feature_store.create_feature_group(
                name="elec_wx_features",
                df=df,
                primary_key=["date_str","sub_region_code"],
                event_time="date",
                description="Engineered electricity demand features",
                online_enabled=True
            )

            logger.info("Feature group created and feature engineering complete.")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self):
        if not self.check_status():
            raise CustomException("Validation failed. Aborting transformation.", sys)
        try:
            df = self.feature_engineering(self.basic_preprocessing())
            df.sort_values("date", inplace=True)

            self.feature_store.create_feature_view(
                name="elec_wx_features_view",
                feature_group_name="elec_wx_features",
                features=[
                    "date", "sub_region_code", "demand", "temperature_2m",
                    "hour", "day_of_week", "month", "is_weekend", "is_holiday"
                ]
            )

            self.feature_store.save_training_dataset(
                feature_view_name="elec_wx_features_view",
                version_description="initial training dataset with all features",
                output_format="csv"
            )

            logger.info("Feature view + training dataset saved successfully.")
            return df
        except Exception as e:
            raise CustomException(e, sys)