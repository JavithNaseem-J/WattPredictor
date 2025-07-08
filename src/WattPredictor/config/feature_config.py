from pathlib import Path
from WattPredictor.entity.config_entity import FeatureStoreConfig
from WattPredictor.utils.helpers import read_yaml, create_directories
from WattPredictor.constants import CONFIG_PATH

class FeatureStoreConfigurationManager:
    def __init__(self, config_filepath: Path = CONFIG_PATH):
        
        create_directories([self.config.artifacts_root])

    def get_feature_store_config(self) -> FeatureStoreConfig:

        config = self.config.feature_store

        feature_store_config = FeatureStoreConfig(
            hopsworks_project_name=config.hopsworks_project_name,
            hopsworks_api_key=config.hopsworks_api_key,
            hopsworks_host=config.hopsworks_host
        )

        return feature_store_config