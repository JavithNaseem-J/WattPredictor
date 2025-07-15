from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class FeatureStoreConfig:
    hopsworks_project_name: str
    hopsworks_api_key: str

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    elec_raw_data: Path
    wx_raw_data: Path
    elec_api: str
    wx_api: str
    elec_api_key: str
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


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    input_seq_len: int
    step_size: int
    n_trials: int
    cutoff_date: str
    model_name: Path


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    cutoff_date: str
    input_seq_len: int
    step_size: int
    img_path: Path
    metrics_path: Path

@dataclass
class DataDriftConfig:
    baseline_start: str
    baseline_end: str
    current_start: str
    current_end: str
    report_dir: Path
