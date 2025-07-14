from WattPredictor.config.data_config import ConfigurationManager
from WattPredictor.config.feature_config import FeatureConfigurationManager
from WattPredictor.components.data_transformation import DataTransformation


class DataTransformationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        feature = FeatureConfigurationManager()
        feature_store_config = feature.get_feature_store_config()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config,feature_store_config=feature_store_config)
        df = data_transformation.transform()

