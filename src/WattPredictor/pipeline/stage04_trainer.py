from WattPredictor.config.model_config import ConfigurationManager
from WattPredictor.config.feature_config import FeatureConfigurationManager
from WattPredictor.components.model_training import ModelTrainer


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        feature = FeatureConfigurationManager()
        feature_store_config = feature.get_feature_store_config()
        data_transformation_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=data_transformation_config,feature_store_config=feature_store_config)
        model_trainer.train()