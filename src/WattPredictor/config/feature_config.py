import os
from pathlib import Path
from WattPredictor.entity.config_entity import FeatureStoreConfig
from WattPredictor.utils.helpers import read_yaml, create_directories
from WattPredictor.constants import *

class FeatureConfigurationManager:
    def __init__(self, 
                 config_filepath=CONFIG_PATH,
                 params_filepath=PARAMS_PATH,
                 schema_filepath=SCHEMA_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        

    def get_feature_store_config(self) -> FeatureStoreConfig:

        config = self.config.feature_store

        feature_store_config = FeatureStoreConfig(
            hopsworks_project_name=config.hopsworks_project_name,
            hopsworks_api_key=os.environ['hopsworks_api_key'],
        )

        return feature_store_config