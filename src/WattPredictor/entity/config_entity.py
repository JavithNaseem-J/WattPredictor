from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, ConfigDict


class IngestionConfig(BaseModel):
    root_dir: Path
    elec_raw_data: Path
    wx_raw_data: Path
    elec_api: str
    wx_api: str
    elec_api_key: str
    data_file: Path
    model_config = ConfigDict(protected_namespaces=())


class ValidationConfig(BaseModel):
    root_dir: Path
    data_file: Path
    status_file: Path
    all_schema: Dict[str, Any]
    model_config = ConfigDict(protected_namespaces=())


class EngineeringConfig(BaseModel):
    root_dir: Path
    data_file: Path
    status_file: Path
    preprocessed: Path
    model_config = ConfigDict(protected_namespaces=())



class TrainerConfig(BaseModel):
    root_dir: Path
    input_seq_len: int
    step_size: int
    cv_folds: int
    model_name: Path
    data_path: Path
    model_config = ConfigDict(protected_namespaces=())


class EvaluationConfig(BaseModel):
    root_dir: Path
    model_path: Path
    input_seq_len: int
    step_size: int
    img_path: Path
    metrics_path: Path
    model_config = ConfigDict(protected_namespaces=())


class PredictionConfig(BaseModel):
    model_path: Path
    predictions_df: Path
    model_config = ConfigDict(protected_namespaces=())


class MonitoringConfig(BaseModel):
    monitoring_df: Path
    model_config = ConfigDict(protected_namespaces=())


class DriftConfig(BaseModel):
    report_dir: Path
    model_config = ConfigDict(protected_namespaces=())
