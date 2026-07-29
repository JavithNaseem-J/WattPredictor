import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

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
        raise ValueError(f"Input DataFrame missing required columns: {missing}")
    if input_seq_len < 672:
        raise ValueError("input_seq_len must be >= 672 for average_demand_last_4_weeks")

    region_codes = ts_data['sub_region_code'].unique()
    features_list = []
    targets_list = []

    additional_cols = ['temperature_2m', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']
    lag_cols = [f'demand_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]

    for code in tqdm(region_codes, desc="Generating TS features (Vectorized)"):
        ts_data_one_location = ts_data.loc[
            ts_data.sub_region_code == code, 
            ['date', 'demand'] + additional_cols
        ].sort_values(by='date')
        
        n_rows = len(ts_data_one_location)
        if n_rows < input_seq_len + 1:
            continue

        demand_arr = ts_data_one_location['demand'].values.astype(np.float64)
        
        # Fast 2D matrix view over sliding windows without array copies
        windows = np.lib.stride_tricks.sliding_window_view(demand_arr, window_shape=input_seq_len)
        
        # Valid window indices
        n_windows = n_rows - input_seq_len
        if n_windows <= 0:
            continue

        valid_indices = np.arange(0, n_windows, step_size)
        if len(valid_indices) == 0:
            continue

        x_demand = windows[valid_indices]
        y_demand = demand_arr[valid_indices + input_seq_len]

        feat_df = pd.DataFrame(x_demand, columns=lag_cols)
        
        for col in additional_cols:
            col_vals = ts_data_one_location[col].values
            feat_df[col] = col_vals[valid_indices + input_seq_len]

        features_list.append(feat_df)
        targets_list.append(pd.Series(y_demand, name='target_demand_next_hour'))

    if not features_list or not targets_list:
        raise ValueError("No valid time-series sequences generated")

    features = pd.concat(features_list, ignore_index=True)
    targets = pd.concat(targets_list, ignore_index=True)

    return features, targets

def average_demand_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    required_columns = [f'demand_previous_{i*7*24}_hour' for i in range(1, 5)]
    if not all(col in X.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain columns {required_columns}")
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
        model = XGBRegressor(**hyperparams, enable_categorical=False, device="cpu", tree_method="hist")
    else:
        raise ValueError("model_type must be 'LightGBM' or 'XGBoost'")
    return Pipeline([
        ('add_average_demand', add_feature_average_demand_last_4_weeks),
        ('model', model)
    ])