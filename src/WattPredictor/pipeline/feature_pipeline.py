from WattPredictor.config.data_config import DataConfigurationManager
from WattPredictor.components.data.ingestion import Ingestion
from WattPredictor.components.data.validation import Validation
from WattPredictor.components.features.engineering import Engineering
from WattPredictor.utils.exception import CustomException



class FeaturePipeline:
    def __init__(self):
        pass

    def run(self):

        config = DataConfigurationManager()

        ingestion_config = config.get_data_ingestion_config()
        ingestor = Ingestion(config=ingestion_config)
        ingestor.download()

        validation_config = config.get_data_validation_config()
        validator = Validation(config=validation_config)
        validator.validator()

        transformation_config = config.get_data_transformation_config()
        transformer = Engineering(config=transformation_config)
        transformer.transform()

