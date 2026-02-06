"""
Unified Configuration Manager for WattPredictor.
Consolidates all configuration management into a single class.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from WattPredictor.entity.config_entity import (
    IngestionConfig, ValidationConfig, EngineeringConfig,
    TrainerConfig, EvaluationConfig,
    PredictionConfig, MonitoringConfig, DriftConfig,
    FeatureStoreConfig
)
from WattPredictor.utils.helpers import read_yaml, create_directories
from WattPredictor.constants.paths import CONFIG_PATH, PARAMS_PATH, SCHEMA_PATH

load_dotenv()


class ConfigManager:
    """
    Unified configuration manager for all pipeline components.
    Replaces: DataConfigurationManager, ModelConfigurationManager,
              InferenceConfigurationManager, FeatureConfigurationManager
    """
    
    def __init__(self, 
                 config_filepath=CONFIG_PATH,
                 params_filepath=PARAMS_PATH,
                 schema_filepath=SCHEMA_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        create_directories([self.config.artifacts_root])

    # ============ Data Pipeline Configs ============
    
    def get_ingestion_config(self) -> IngestionConfig:
        """Config for data ingestion from APIs."""
        config = self.config.data
        create_directories([config.root_dir])
        
        return IngestionConfig(
            root_dir=Path(config.root_dir),
            elec_raw_data=Path(config.elec_raw_data),
            wx_raw_data=Path(config.wx_raw_data),
            elec_api=os.environ.get('ELEC_API', ''),
            wx_api=os.environ.get('WX_API', ''),
            elec_api_key=os.environ.get('ELEC_API_KEY', ''),
            data_file=Path(config.data_file)
        )
    
    def get_validation_config(self) -> ValidationConfig:
        """Config for data validation."""
        config = self.config.validation
        schema = self.schema.columns
        create_directories([config.root_dir])
        
        return ValidationConfig(
            root_dir=config.root_dir,
            status_file=config.status_file,
            data_file=config.data_file,
            all_schema=schema,
        )
    
    def get_engineering_config(self) -> EngineeringConfig:
        """Config for feature engineering."""
        config = self.config.engineering
        create_directories([config.root_dir])
        
        return EngineeringConfig(
            root_dir=Path(config.root_dir),
            status_file=Path(config.status_file),
            data_file=config.data_file,
            preprocessed=Path(config.preprocessed)
        )

    # ============ Model Pipeline Configs ============
    
    def get_trainer_config(self) -> TrainerConfig:
        """Config for model training."""
        config = self.config.trainer
        params = self.params.training
        create_directories([config.root_dir])
        
        return TrainerConfig(
            root_dir=Path(config.root_dir),
            input_seq_len=params.input_seq_len,
            step_size=params.step_size,
            cv_folds=params.cv_folds,
            model_name=Path(config.model_name),
            data_path=Path(config.data_path)
        )
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Config for model evaluation."""
        config = self.config.evaluation
        params = self.params.training
        create_directories([config.root_dir])
        
        return EvaluationConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            input_seq_len=params.input_seq_len,
            step_size=params.step_size,
            img_path=Path(config.img_path),
            metrics_path=Path(config.metrics_path)
        )

    # ============ Inference Pipeline Configs ============
    
    def get_prediction_config(self) -> PredictionConfig:
        """Config for batch predictions."""
        config = self.config.prediction
        
        return PredictionConfig(
            model_name=config.model_name,
            model_version=config.model_version,
            feature_view_name=config.feature_view_name,
            feature_view_version=config.feature_view_version,
            n_features=config.n_features,
            predictions_df=Path(config.predictions_df),
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Config for monitoring predictions vs actuals."""
        config = self.config.monitoring
        
        return MonitoringConfig(
            predictions_fg_name=config.predictions_fg_name,
            predictions_fg_version=config.predictions_fg_version,
            actuals_fg_name=config.actuals_fg_name,
            actuals_fg_version=config.actuals_fg_version,
            monitoring_df=Path(config.monitoring_df)
        )
    
    def get_drift_config(self) -> DriftConfig:
        """Config for drift detection."""
        config = self.config.drift
        create_directories([config.root_dir])
        
        return DriftConfig(
            report_dir=Path(config.report_dir)
        )

    # ============ Feature Store Config ============
    
    def get_feature_store_config(self) -> FeatureStoreConfig:
        """Config for Hopsworks feature store."""
        config = self.config.feature_store
        
        return FeatureStoreConfig(
            hopsworks_project_name=config.hopsworks_project_name,
            hopsworks_api_key=os.environ.get('HOPSWORKS_API_KEY', ''),
        )


# Legacy aliases for backward compatibility
# These can be removed once all imports are updated
DataConfigurationManager = ConfigManager
ModelConfigurationManager = ConfigManager
InferenceConfigurationManager = ConfigManager
FeatureConfigurationManager = ConfigManager
