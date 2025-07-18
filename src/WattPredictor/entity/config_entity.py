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
    start_date: str
    end_date: str

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
    label_encoder: Path

@dataclass
class TrainerConfig:
    root_dir: Path
    input_seq_len: int
    step_size: int
    n_trials: int
    cutoff_date: str
    model_name: Path

@dataclass
class EvaluationConfig:
    root_dir: Path
    model_path: Path
    cutoff_date: str
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

@dataclass
class MonitoringConfig:
  predictions_fg_name: str
  predictions_fg_version: int
  actuals_fg_name: str
  actuals_fg_version: int

@dataclass
class DriftConfig:
    baseline_start: str
    baseline_end: str
    current_start: str
    current_end: str
    report_dir: Path
