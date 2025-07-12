import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import Tuple


def get_cutoff_indices_features_and_target(data: pd.DataFrame,input_seq_len: int,step_size: int) -> list:
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


def features_and_target(ts_data: pd.DataFrame,input_seq_len: int,step_size: int):

    assert set(ts_data.columns) == {'date', 'demand', 'sub_region_code', 'temperature_2m'}

    region_codes = ts_data['sub_region_code'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()

    for code in tqdm(region_codes, desc="Generating TS features"):
        ts_data_one_location = ts_data.loc[
            ts_data.sub_region_code == code, 
            ['date', 'temperature_2m', 'demand']
        ].sort_values(by='date')

        indices = get_cutoff_indices_features_and_target(
            ts_data_one_location, input_seq_len, step_size
        )

        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float64)
        y = np.ndarray(shape=(n_examples), dtype=np.float64)
        date_hours = []
        temperatures = []

        for i, (start, mid, end) in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[start:mid]['demand'].values
            y[i] = ts_data_one_location.iloc[mid]['demand']
            date_hours.append(ts_data_one_location.iloc[mid]['date'])
            temperatures.append(ts_data_one_location.iloc[mid]['temperature_2m'])

        features_one_location = pd.DataFrame(
            x,
            columns=[f'demand_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['date'] = date_hours
        features_one_location['sub_region_code'] = code
        features_one_location['temperature_2m'] = temperatures

        targets_one_location = pd.DataFrame(y, columns=['target_demand_next_hour'])

        features = pd.concat([features, features_one_location], ignore_index=True)
        targets = pd.concat([targets, targets_one_location], ignore_index=True)

    return features, targets['target_demand_next_hour']


def ts_train_test_split(df: pd.DataFrame,cutoff_date: datetime,target_column_name: str,):
    df['date'] = pd.to_datetime(df['date'], utc=True)

    train_data = df[df['date'] < cutoff_date].reset_index(drop=True)
    test_data = df[df['date'] >= cutoff_date].reset_index(drop=True)

    X_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]
    X_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]

    return X_train, y_train, X_test, y_test
