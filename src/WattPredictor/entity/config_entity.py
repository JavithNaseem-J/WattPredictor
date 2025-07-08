from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    elec_raw_data: Path
    wx_raw_data: Path
    data_file: Path
    start_date: str
    end_date: str

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_file: Path
    status_file: Path
    all_schema: dict

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_file: Path
    status_file: str
    label_encoder: Path
    x_transform: Path
    y_transform: Path
    train_features: Path
    test_features: Path
    train_target: Path
    test_target: Path
    input_seq_len: int
    step_size: int
    cutoff_date: str

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    x_transform: Path
    y_transform: Path
    model_name: str
    scoring: str
    cv_folds: int
    n_jobs: int
    n_trials: int
    early_stopping_rounds: int

@dataclass
class ModelEvaluationConfig:
    model_path: Path
    x_transform: Path
    y_transform: Path
    metrics_path: Path