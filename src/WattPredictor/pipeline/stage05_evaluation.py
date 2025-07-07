from WattPredictor.config.config import ConfigurationManager
from WattPredictor.components.model_evaluation import ModelEvaluation


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate()