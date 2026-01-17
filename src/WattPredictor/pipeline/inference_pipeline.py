from datetime import datetime
from WattPredictor.config.inference_config import InferenceConfigurationManager
from WattPredictor.entity.config_entity import PredictionConfig
from WattPredictor.components.inference.predictor import Predictor
from WattPredictor.utils.exception import CustomException



class InferencePipeline:
    def __init__(self):
        pass

    def run(self):
        config = InferenceConfigurationManager()
        predictor_config = config.get_data_prediction_config()
        predictor = Predictor(config=predictor_config)
        predictor.predict(save_to_store=True)


if __name__ == "__main__":
    pipeline = InferencePipeline()
    pipeline.run()

