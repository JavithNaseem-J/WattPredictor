from WattPredictor.config.data_config import ConfigurationManager
from WattPredictor.components.data_validation import DataValidation


class DataValidationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(data_validation_config)
        data_validation.validation()