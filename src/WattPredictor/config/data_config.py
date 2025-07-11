import os
from pathlib import Path
from WattPredictor.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from WattPredictor.utils.helpers import read_yaml, create_directories
from WattPredictor.constants import CONFIG_PATH, PARAMS_PATH, SCHEMA_PATH


class ConfigurationManager:
    def __init__(self, 
                 config_filepath=CONFIG_PATH,
                 params_filepath=PARAMS_PATH,
                 schema_filepath=SCHEMA_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        params = self.params.dates

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
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
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.columns
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            status_file=config.status_file,
            data_file=config.data_file,
            all_schema=schema,
        )
        return data_validation_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        schema = self.schema
        params = self.params.transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_file=Path(config.data_file),
            status_file=Path(config.status_file),
            label_encoder=Path(config.label_encoder),
            train_features=Path(config.train_features),
            test_features=Path(config.test_features),
            input_seq_len=params.input_seq_len,
            step_size=params.step_size,
            cutoff_date=params.cutoff_date
        )

        return data_transformation_config