import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from WattPredictor.entity.config_entity import EngineeringConfig
from WattPredictor.utils.feature import feature_store_instance
from WattPredictor.utils.helpers import create_directories, save_bin
from WattPredictor.utils.exception import CustomException
from WattPredictor.utils.logging import logger

class Engineering:
    def __init__(self, config: EngineeringConfig):
        self.config = config
        self.feature_store = feature_store_instance()

    def check_status(self):
        with open(self.config.status_file, 'r') as f:
            status_data = json.load(f)
        return status_data.get("validation_status", False)

    def basic_preprocessing(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_file)
        le = LabelEncoder()
        df['sub_region_code'] = le.fit_transform(df['subba'])
        df.rename(columns={'subba': 'sub_region', 'value': 'demand'}, inplace=True)
        df = df[['date_str', 'date', 'sub_region_code', 'demand', 'temperature_2m']]
        create_directories([Path(self.config.label_encoder).parent])
        save_bin(le, self.config.label_encoder)
        self.feature_store.upload_file_safely(self.config.label_encoder, "label_encoder.pkl")
        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df['hour'] = df['date'].dt.hour.astype('int64')
        df['day_of_week'] = df['date'].dt.dayofweek.astype('int64')
        df['month'] = df['date'].dt.month.astype('int64')
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype('int64')
        holidays = calendar().holidays(start=df['date'].min(), end=df['date'].max())
        df['is_holiday'] = df['date'].isin(holidays).astype('int64')
        df['temperature_2m'] = df['temperature_2m'].astype('float64')
        df['demand'] = df['demand'].astype('float64')
        self.feature_store.create_feature_group(
            name="elec_wx_features",
            df=df,
            primary_key=["date_str", "sub_region_code"],
            event_time="date",
            description="Engineered electricity demand features",
            online_enabled=True
        )
        logger.info("Feature group 'elec_wx_features' created successfully")
        return df

    def transform(self):
        if not self.check_status():
            raise CustomException("Validation failed. Aborting transformation.", None)
        df = self.feature_engineering(self.basic_preprocessing())
        df.sort_values("date", inplace=True)
        self.feature_store.create_feature_view(
            name="elec_wx_features_view",
            feature_group_name="elec_wx_features",
            features=["date", "sub_region_code", "demand", "temperature_2m",
                      "hour", "day_of_week", "month", "is_weekend", "is_holiday"]
        )
        self.feature_store.save_training_dataset(
            feature_view_name="elec_wx_features_view",
            version_description="Training dataset with essential features for electricity demand prediction",
            output_format="csv"
        )
        logger.info("Feature view 'elec_wx_features_view' created successfully")
        return df