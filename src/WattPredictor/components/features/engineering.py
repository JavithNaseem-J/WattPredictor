import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from WattPredictor.entity.config_entity import EngineeringConfig
from WattPredictor.utils.helpers import create_directories
from WattPredictor.utils.logging import logger


class Engineering:
 
    
    def __init__(self, config: EngineeringConfig):
        self.config = config

    def basic_preprocessing(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_file)
        le = LabelEncoder()
        df['sub_region_code'] = le.fit_transform(df['subba'])
        df.rename(columns={'subba': 'sub_region', 'value': 'demand'}, inplace=True)
        df = df[['date_str', 'date', 'sub_region_code', 'demand', 'temperature_2m']]
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
        
        logger.info("Feature engineering completed")
        return df

    def transform(self):
        # DVC handles dependency checking - no need to manually check validation status
        df = self.feature_engineering(self.basic_preprocessing())
        df.sort_values("date", inplace=True)
        
        create_directories([self.config.preprocessed.parent])
        df.to_csv(self.config.preprocessed, index=False)
        logger.info(f"Preprocessed data saved to {self.config.preprocessed}")
        
        return df