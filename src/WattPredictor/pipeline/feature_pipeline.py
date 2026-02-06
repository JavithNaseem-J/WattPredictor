from WattPredictor.config.config_manager import ConfigManager
from WattPredictor.components.features.ingestion import Ingestion
from WattPredictor.components.features.validation import Validation
from WattPredictor.components.features.engineering import Engineering


class FeaturePipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigManager()

        ingestion_config = config.get_ingestion_config()
        ingestor = Ingestion(config=ingestion_config)
        ingestor.download()

        validation_config = config.get_validation_config()
        validator = Validation(config=validation_config)
        validator.validator()

        transformation_config = config.get_engineering_config()
        transformer = Engineering(config=transformation_config)
        transformer.transform()


if __name__ == "__main__":
    pipeline = FeaturePipeline()
    pipeline.run()
