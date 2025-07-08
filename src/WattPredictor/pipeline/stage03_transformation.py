from WattPredictor.config.data_config import ConfigurationManager
from WattPredictor.components.data_transformation import DataTransformation


class DataTransformationPipeline:
    def __init__(self):
        pass

    def run(self):
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            train_df, test_df = data_transformation.train_test_splitting()
            (train_x, train_y), (test_x, test_y) = data_transformation.preprocess_features(train_df, test_df)