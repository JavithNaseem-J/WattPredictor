import os
from pathlib import Path
from WattPredictor.entity.config_entity import IngestionConfig, ValidationConfig, EngineeringConfig
from WattPredictor.utils.helpers import read_yaml, create_directories
from WattPredictor.constants.paths import CONFIG_PATH, PARAMS_PATH, SCHEMA_PATH


class DataConfigurationManager:
    def __init__(self, 
                 config_filepath=CONFIG_PATH,
                 params_filepath=PARAMS_PATH,
                 schema_filepath=SCHEMA_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> IngestionConfig:
        config = self.config.data_ingestion
        params = self.params.dates

        create_directories([config.root_dir])

        data_ingestion_config = IngestionConfig(
            root_dir=Path(config.root_dir),
            elec_raw_data=Path(config.elec_raw_data),
            wx_raw_data=Path(config.wx_raw_data),
            elec_api= os.environ['elec_api'],
            wx_api= os.environ['wx_api'],
            elec_api_key= os.environ['elec_api_key'],
            data_file=Path(config.data_file),
            start_date=params.start_date,
            end_date=params.end_date
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> ValidationConfig:
        config = self.config.data_validation
        schema = self.schema.columns
        
        create_directories([config.root_dir])
        
        data_validation_config = ValidationConfig(
            root_dir=config.root_dir,
            status_file=config.status_file,
            data_file=config.data_file,
            all_schema=schema,
        )
        
        return data_validation_config
    

    def get_data_transformation_config(self) -> EngineeringConfig:
        config = self.config.data_transformation
        params = self.params.training

        create_directories([config.root_dir])

        data_transformation_config = EngineeringConfig(
            root_dir=Path(config.root_dir),
            data_file=Path(config.data_file),
            status_file=Path(config.status_file),
            label_encoder=Path(config.label_encoder)
        )

        return data_transformation_config
    
    def get_data_drift_config(self) -> DriftConfig:
        config = self.config.data_drift
        params = self.params.drift

        create_directories([config.root_dir])
        
        data_drift_cofig =  DriftConfig(
            baseline_start=self.params.drift.baseline_start,
            baseline_end=self.params.drift.baseline_end,
            current_start=self.params.drift.current_start,
            current_end=self.params.drift.current_end,
            report_dir=Path(config.report_dir)
        )
        
        return data_drift_cofig