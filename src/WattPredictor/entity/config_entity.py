from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field


class FeatureStoreConfig(BaseModel):
    hopsworks_project_name: str
    hopsworks_api_key: str


class IngestionConfig(BaseModel):
    root_dir: Path
    elec_raw_data: Path
    wx_raw_data: Path
    elec_api: str
    wx_api: str
    elec_api_key: str
    data_file: Path



class ValidationConfig(BaseModel):
    root_dir: Path
    data_file: Path
    status_file: Path
    all_schema: Dict[str, Any]




class EngineeringConfig(BaseModel):
    root_dir: Path
    data_file: Path
    status_file: str
    preprocessed: Path


class TrainerConfig(BaseModel):
    root_dir: Path
    input_seq_len: int
    step_size: int
    cv_folds: int
    model_name: Path
    data_path: Path


class EvaluationConfig(BaseModel):
    root_dir: Path
    model_path: Path
    input_seq_len: int
    step_size: int
    img_path: Path
    metrics_path: Path


class PredictionConfig(BaseModel):
    model_name: str
    model_version: int
    feature_view_name: str
    feature_view_version: int
    n_features: int
    predictions_df: Path


class MonitoringConfig(BaseModel):
    predictions_fg_name: str
    predictions_fg_version: int
    actuals_fg_name: str
    actuals_fg_version: int
    monitoring_df: Path


class DriftConfig(BaseModel):
    report_dir: Path
