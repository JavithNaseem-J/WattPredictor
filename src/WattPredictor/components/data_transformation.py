import os
import sys
import tqdm
import json
import joblib
import numpy as np
from WattPredictor import logger
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from WattPredictor.entity.config_entity import DataTransformationConfig
from WattPredictor.utils.helpers import create_directories, save_bin, load_json
from WattPredictor.utils.exception import CustomException
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def check_status(self):
        try:
            with open(self.config.status_file, 'r') as f:
                status_data = json.load(f)
            validation_status = status_data.get("validation_status", False)
            logger.info(f"Data validation status: {validation_status}")
            return validation_status
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in status file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error reading validation status: {e}")
            return False


    def basic_preprocessing(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.config.data_file)
            df = df[['period', 'subba', 'value', 'temperature_2m']]
            le = LabelEncoder()
            df['sub_region_code'] = le.fit_transform(df['subba'])

            df.rename(columns={
                'period': 'date',
                'subba': 'sub_region',
                'value': 'demand'
            }, inplace=True)

            df = df[['date', 'sub_region_code', 'demand', 'temperature_2m']]

            create_directories([os.path.dirname(self.config.label_encoder)])
            save_bin(le, self.config.label_encoder)

            logger.info("Basic preprocessing completed.")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

            holidays = calendar().holidays(start=df['date'].min(), end=df['date'].max())
            df['is_holiday'] = df['date'].isin(holidays).astype(int)

            logger.info("Feature engineering completed.")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def train_test_splitting(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            df = self.feature_engineering(self.basic_preprocessing())
            df.sort_values("date", inplace=True)

            cutoff = pd.to_datetime(self.config.cutoff_date, utc=True)

            train_df = df[df['date'] < cutoff].reset_index(drop=True)
            test_df = df[df['date'] >= cutoff].reset_index(drop=True)

            logger.info(f"Train size: {train_df.shape}, Test size: {test_df.shape}")
            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)
        
    def _get_cutoff_indices(self, df: pd.DataFrame, input_seq_len: int, step_size: int):
        stop = len(df) - input_seq_len - 1
        return [(i, i + input_seq_len, i + input_seq_len + 1) for i in range(0, stop, step_size)]
        
    def transform_ts_data_into_features_and_target(self, ts_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
            
            assert set(['date', 'demand', 'sub_region_code', 'temperature_2m']).issubset(ts_data.columns)

            region_codes = ts_data['sub_region_code'].unique()
            features = pd.DataFrame()
            targets = pd.DataFrame()

            input_seq_len = self.config.input_seq_len
            step_size = self.config.step_size

            for code in tqdm.tqdm(region_codes, desc="Transforming TS Data"):
                ts_one = ts_data[ts_data['sub_region_code'] == code].sort_values(by='date')
                indices = self._get_cutoff_indices(ts_one, input_seq_len, step_size)

                x = np.zeros((len(indices), input_seq_len), dtype=np.float64)
                y = np.zeros((len(indices)), dtype=np.float64)
                date_hours, temps = [], []

                for i, (start, mid, end) in enumerate(indices):
                    x[i, :] = ts_one.iloc[start:mid]['demand'].values
                    y[i] = ts_one.iloc[mid]['demand']
                    date_hours.append(ts_one.iloc[mid]['date'])
                    temps.append(ts_one.iloc[mid]['temperature_2m'])

                features_one = pd.DataFrame(
                    x,
                    columns=[f'demand_prev_{i+1}_hr' for i in reversed(range(input_seq_len))]
                )
                features_one['date'] = date_hours
                features_one['sub_region_code'] = code
                features_one['temperature_2m'] = temps

                targets_one = pd.DataFrame(y, columns=['target_demand_next_hour'])

                features = pd.concat([features, features_one], ignore_index=True)
                targets = pd.concat([targets, targets_one], ignore_index=True)

            return features, targets['target_demand_next_hour']

    def preprocess_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        try:
            validation_status = self.check_status()

            if not validation_status:
                logger.error("Data validation failed. Aborting data transformation.")
                return None

            logger.info("Validation PASSED. Proceeding with feature transformation...")

            # Transform to supervised format
            train_x, train_y = self.transform_ts_data_into_features_and_target(train_df)
            test_x, test_y = self.transform_ts_data_into_features_and_target(test_df)

            # Drop 'date' before saving as .npy
            train_x_npy = train_x.drop(columns=["date"], errors="ignore")
            test_x_npy = test_x.drop(columns=["date"], errors="ignore")

            # Save numpy arrays
            np.save(self.config.x_transform, train_x_npy.values)
            np.save(self.config.y_transform, train_y.values)

            logger.info(f"Shapes - Train X: {train_x_npy.shape}, Train Y: {train_y.shape}, test X: {test_x_npy.shape}, Test Y: {test_y.shape}")

            # Save CSVs for inspection
            train_x.to_csv(self.config.train_features, index=False)
            train_y.to_csv(self.config.train_target, index=False)
            test_x.to_csv(self.config.test_features, index=False)
            test_y.to_csv(self.config.test_target, index=False)

            logger.info("Feature transformation and saving completed.")
            return (train_x, train_y), (test_x, test_y)

        except Exception as e:
            raise CustomException(e, sys)