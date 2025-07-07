from WattPredictor.config.config import ConfigurationManager
from WattPredictor.components.data_ingestion import DataIngestion


class DataIngestionPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        ingestion_config = config.get_data_ingestion_config()
        ingestion = DataIngestion(config=ingestion_config)
        df = ingestion.download()