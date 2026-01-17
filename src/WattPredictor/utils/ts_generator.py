import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import Tuple
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from WattPredictor.utils.exception import CustomException

def get_cutoff_indices_features_and_target(data: pd.DataFrame, input_seq_len: int, step_size: int) -> list:
    stop_position = len(data) - 1
    subseq_first_idx = 0
    subseq_mid_idx = input_seq_len
    subseq_last_idx = input_seq_len + 1
    indices = []
    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size
    return indices

def features_and_target(ts_data: pd.DataFrame, input_seq_len: int, step_size: int):
    required_columns = {'date', 'demand', 'sub_region_code', 'temperature_2m', 
                        'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday'}
    if not required_columns.issubset(ts_data.columns):
        missing = required_columns - set(ts_data.columns)
        raise CustomException(f"Input DataFrame missing required columns: {missing}", None)
    if input_seq_len < 672:
        raise CustomException("input_seq_len must be >= 672 for average_demand_last_4_weeks", None)

    region_codes = ts_data['sub_region_code'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()

    for code in tqdm(region_codes, desc="Generating TS features"):
        ts_data_one_location = ts_data.loc[
            ts_data.sub_region_code == code, 
            ['date', 'demand', 'temperature_2m', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']
        ].sort_values(by='date')
        if len(ts_data_one_location) < input_seq_len + 1:
            continue
        indices = get_cutoff_indices_features_and_target(ts_data_one_location, input_seq_len, step_size)
        if not indices:
            continue

        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float64)
        y = np.ndarray(shape=(n_examples), dtype=np.float64)
        additional_features = {
            'temperature_2m': [], 'hour': [], 'day_of_week': [], 'month': [], 
            'is_weekend': [], 'is_holiday': []
        }

        for i, (start, mid, end) in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[start:mid]['demand'].values
            y[i] = ts_data_one_location.iloc[mid]['demand']
            for col in additional_features:
                additional_features[col].append(ts_data_one_location.iloc[mid][col])

        features_one_location = pd.DataFrame(
            x, columns=[f'demand_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        for col in additional_features:
            features_one_location[col] = additional_features[col]
        features_one_location = features_one_location.select_dtypes(include=['int64', 'float64', 'bool'])
        targets_one_location = pd.DataFrame(y, columns=['target_demand_next_hour'])

        features = pd.concat([features, features_one_location], ignore_index=True)
        targets = pd.concat([targets, targets_one_location], ignore_index=True)

    if features.empty or targets.empty:
        raise CustomException("No valid time-series sequences generated", None)
    return features, targets['target_demand_next_hour']

def ts_train_test_split(df: pd.DataFrame, cutoff_date: datetime, target_column_name: str):
    df['date'] = pd.to_datetime(df['date'], utc=True)
    train_data = df[df['date'] < cutoff_date].reset_index(drop=True)
    test_data = df[df['date'] >= cutoff_date].reset_index(drop=True)
    X_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]
    X_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]
    return X_train, y_train, X_test, y_test

def average_demand_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    required_columns = [f'demand_previous_{i*7*24}_hour' for i in range(1, 5)]
    if not all(col in X.columns for col in required_columns):
        raise CustomException(f"Input DataFrame must contain columns {required_columns}", None)
    X['average_demand_last_4_weeks'] = 0.25 * (
        X[f'demand_previous_{7*24}_hour'] + 
        X[f'demand_previous_{2*7*24}_hour'] + 
        X[f'demand_previous_{3*7*24}_hour'] + 
        X[f'demand_previous_{4*7*24}_hour']
    )
    return X

def get_pipeline(model_type: str, **hyperparams) -> Pipeline:
    add_feature_average_demand_last_4_weeks = FunctionTransformer(
        average_demand_last_4_weeks, validate=False)
    if model_type == "LightGBM":
        model = LGBMRegressor(**hyperparams, verbosity=-1)
    elif model_type == "XGBoost":
        # Force CPU to avoid CUDA errors
        model = XGBRegressor(**hyperparams, enable_categorical=False, device="cpu", tree_method="hist")
    else:
        raise ValueError("model_type must be 'LightGBM' or 'XGBoost'")
    return Pipeline([
        ('add_average_demand', add_feature_average_demand_last_4_weeks),
        ('model', model)
    ])