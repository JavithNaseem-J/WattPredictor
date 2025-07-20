import os
from pathlib import Path
from datetime import datetime as dt, timezone
from WattPredictor.entity.config_entity import PredictionConfig, MonitoringConfig, DriftConfig
from WattPredictor.utils.helpers import read_yaml, create_directories
from WattPredictor.constants.paths import CONFIG_PATH, PARAMS_PATH, SCHEMA_PATH



class InferenceConfigurationManager:
    def __init__(self, 
                 config_filepath=CONFIG_PATH,
                 params_filepath=PARAMS_PATH,
                 schema_filepath=SCHEMA_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_prediction_config(self) -> PredictionConfig:
        config = self.config.prediction
        
        data_prediction_config =  PredictionConfig(
            model_name = config.model_name,
            model_version= config.model_version,
            feature_view_name= config.feature_view_name,
            feature_view_version = config.feature_view_version,
            n_features = config.n_features,
            predictions_df=Path(config.predictions_df),
        )
        
        return data_prediction_config
    

    def get_data_monitoring_config(self) -> MonitoringConfig:
        config = self.config.monitoring
        
        data_monitoring_config =  MonitoringConfig(
            predictions_fg_name= config.predictions_fg_name,
            predictions_fg_version= config.predictions_fg_version,
            actuals_fg_name= config.actuals_fg_name,
            actuals_fg_version= config.actuals_fg_version,
            monitoring_df = Path(config.monitoring_df)
        )
        
        return data_monitoring_config


    def get_data_drift_config(self) -> DriftConfig:
        config = self.config.drift

        create_directories([config.root_dir])
        
        data_drift_cofig =  DriftConfig(
            report_dir=Path(config.report_dir)
        )
        
        return data_drift_cofig