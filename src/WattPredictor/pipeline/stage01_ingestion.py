from WattPredictor.config.data_config import ConfigurationManager
from WattPredictor.config.feature_config import FeatureConfigurationManager
from WattPredictor.components.data_ingestion import DataIngestion

class DataIngestionPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        feature = FeatureConfigurationManager()
        ingestion_config = config.get_data_ingestion_config()
        feature_store_config = feature.get_feature_store_config()
        ingestion = DataIngestion(config=ingestion_config, feature_store_config=feature_store_config)
        df = ingestion.download()