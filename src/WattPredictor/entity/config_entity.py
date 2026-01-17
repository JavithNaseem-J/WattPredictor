from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class FeatureStoreConfig:
    hopsworks_project_name: str
    hopsworks_api_key: str

@dataclass(frozen=True)
class IngestionConfig:
    root_dir: Path
    elec_raw_data: Path
    wx_raw_data: Path
    elec_api: str
    wx_api: str
    elec_api_key: str
    data_file: Path
    

@dataclass(frozen=True)
class ValidationConfig:
    root_dir: Path
    data_file: Path
    status_file: Path
    all_schema: dict

@dataclass
class EngineeringConfig:
    root_dir: Path
    data_file: Path
    status_file: str
    preprocessed: Path

@dataclass
class TrainerConfig:
    root_dir: Path
    input_seq_len: int
    step_size: int
    cv_folds: int
    model_name: Path
    data_path: Path

@dataclass
class EvaluationConfig:
    root_dir: Path
    model_path: Path
    input_seq_len: int
    step_size: int
    img_path: Path
    metrics_path: Path

@dataclass
class PredictionConfig:
    model_name: str
    model_version: int
    feature_view_name: str
    feature_view_version: int
    n_features: int
    predictions_df: Path

@dataclass
class MonitoringConfig:
    predictions_fg_name: str
    predictions_fg_version: int
    actuals_fg_name: str
    actuals_fg_version: int
    monitoring_df: Path

@dataclass
class DriftConfig:
    report_dir: Path
