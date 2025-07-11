from WattPredictor.config.model_config import ConfigurationManager
from WattPredictor.config.feature_config import FeatureConfigurationManager
from WattPredictor.components.model_evaluation import ModelEvaluation


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        feature = FeatureConfigurationManager()
        feature_store_config = feature.get_feature_store_config()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config, feature_store_config=feature_store_config)
        model_evaluation.evaluate()